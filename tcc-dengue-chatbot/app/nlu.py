"""NLU híbrido para interpretação de perguntas epidemiológicas."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

from openai import OpenAI

ALLOWED_INTENTS = {
    "incidencia",
    "total",
    "ranking",
    "comparacao",
    "tendencia",
    "media_estado",
    "pico",
    "desconhecida",
}

UF_SIGLAS = {
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
}

STATE_NAME_TO_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}

OPENAI_TIMEOUT_S = 20.0


def _strip_accents(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def _norm(value: str) -> str:
    return _strip_accents(str(value or "").strip().lower())


def _dedupe_strs(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw or "").strip()
        if not s:
            continue
        key = _norm(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _default_interpretation() -> dict[str, Any]:
    return {
        "intencao": "desconhecida",
        "municipios": [],
        "ufs": [],
        "ano": None,
        "semana": None,
        "metrica": "casos",
        "periodo": "anual",
        "limite": 10,
    }


def _extract_year(question: str) -> int | None:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", _norm(question))
    return int(m.group(1)) if m else None


def _extract_week(question: str) -> int | None:
    text = _norm(question)
    for pattern in (
        r"(?:semana(?: epidemiologica)?)\s*[:\-]?\s*(\d{1,2})\b",
        r"\bsem\s*[:\-]?\s*(\d{1,2})\b",
    ):
        m = re.search(pattern, text)
        if m:
            week = int(m.group(1))
            if 1 <= week <= 53:
                return week
    return None


def _extract_limit(question: str) -> int:
    text = _norm(question)
    m = re.search(r"\b(?:top|limite|mostrar|mostre)\s+(\d{1,2})\b", text)
    if m:
        return max(1, min(int(m.group(1)), 10))
    return 10


def _extract_ufs(question: str) -> list[str]:
    text_norm = _norm(question)
    ufs: list[str] = []

    for state_name, uf in STATE_NAME_TO_UF.items():
        if re.search(rf"\b{re.escape(state_name)}\b", text_norm):
            ufs.append(uf)

    upper = str(question or "").upper()
    for uf in sorted(UF_SIGLAS):
        if re.search(rf"\b{uf}\b", upper):
            ufs.append(uf)
    return _dedupe_strs(ufs)


def _split_location_and_uf(location: str) -> tuple[str, str | None]:
    raw = str(location or "").strip(" .,:;?!")
    if not raw:
        return "", None

    parts = raw.split()
    if len(parts) >= 2 and parts[-1].upper() in UF_SIGLAS:
        return " ".join(parts[:-1]).strip(), parts[-1].upper()

    norm_raw = _norm(raw)
    for state_name, uf in sorted(STATE_NAME_TO_UF.items(), key=lambda item: len(item[0]), reverse=True):
        if norm_raw.endswith(" " + state_name):
            nome = raw[: len(raw) - len(state_name)].strip(" ,-/")
            return nome, uf
        if norm_raw == state_name:
            return "", uf
    return raw, None


def _clean_location_phrase(location: str) -> str:
    value = str(location or "").strip(" .,:;?!")
    value = re.sub(
        r"\b(?:de dengue|nas ultimas semanas|nas últimas semanas|ultimas semanas|últimas semanas)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s{2,}", " ", value).strip(" .,:;?!")
    return value


def _extract_compare_municipios(question: str) -> list[str]:
    text = str(question or "").strip()
    m = re.search(
        r"\b(?:compare|comparar|comparacao|comparação)\b\s+(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    chunk = _clean_location_phrase(m.group(1))
    chunk = re.sub(r"\bcom\b", " e ", chunk, flags=re.IGNORECASE)
    parts = re.split(r"\s+e\s+|,\s*", chunk)
    out: list[str] = []
    for part in parts:
        name, _ = _split_location_and_uf(_clean_location_phrase(part))
        if len(name) >= 2:
            out.append(name)
    return _dedupe_strs(out)


def _extract_primary_location(question: str) -> tuple[str | None, str | None]:
    text = str(question or "").strip()

    patterns = [
        r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]{1,60}?)\s+esta\b",
        r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]{1,60}?)\s+está\b",
        r"\b(?:em|no|na)\s+(.+?)\s+em\s+(?:19\d{2}|20\d{2})\b",
        r"\b(?:em|no|na)\s+(.+?)(?:\s+nas?\s+ultimas?\s+semanas|\s+nas?\s+últimas?\s+semanas|\?|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        location = _clean_location_phrase(m.group(1))
        if not location:
            continue
        if re.fullmatch(r"\d{4}", location):
            continue
        if _norm(location) in {"a dengue", "dengue"}:
            continue
        if location.upper() in UF_SIGLAS or _norm(location) in STATE_NAME_TO_UF:
            continue
        municipio, uf = _split_location_and_uf(location)
        municipio = _clean_location_phrase(municipio)
        if municipio:
            return municipio, uf
    return None, None


def _detect_intent(question: str) -> str:
    text = _norm(question)

    if any(key in text for key in ["semana com mais casos", "pico epidemiologico", "pico epidemiológico", "pico de casos"]):
        return "pico"
    if any(key in text for key in ["acima da media", "abaixo da media", "acima da média", "abaixo da média", "media do", "média do", "media estadual", "média estadual"]):
        return "media_estado"
    if any(key in text for key in ["compare", "comparar", "comparacao", "comparação"]):
        return "comparacao"
    if any(key in text for key in ["aumentando", "tendencia", "tendência", "ultimas semanas", "últimas semanas"]):
        return "tendencia"
    if any(key in text for key in ["ranking", "maior incidencia", "maior incidência", "maiores incidencias", "maiores incidências", "maior incid", "mais casos", "quais cidades", "quais municipios", "quais municípios"]):
        return "ranking"
    if any(key in text for key in ["incidencia", "incidência", "por 100 mil", "100 mil"]):
        return "incidencia"
    if any(key in text for key in ["quantos casos", "total de casos", "casos de dengue", "quantos"]):
        return "total"
    return "desconhecida"


def _detect_metric(question: str, intent: str) -> str:
    text = _norm(question)
    if intent in {"incidencia", "media_estado"}:
        return "incidencia"
    if intent == "comparacao":
        return "incidencia" if "incid" in text else "incidencia"
    if intent == "ranking":
        return "incidencia" if "incid" in text else "casos"
    if any(key in text for key in ["incidencia", "incidência", "100 mil"]):
        return "incidencia"
    return "casos"


def _detect_period(question: str, intent: str) -> str:
    text = _norm(question)
    if intent in {"tendencia", "pico"}:
        return "semanal"
    if any(key in text for key in ["semanal", "semana", "ultimas semanas", "últimas semanas"]):
        return "semanal"
    return "anual"


def _normalize_interpretation(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = _default_interpretation()
    if raw:
        data.update(raw)

    intent = str(data.get("intencao") or "desconhecida").strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "desconhecida"
    data["intencao"] = intent

    municipios = data.get("municipios") or []
    if isinstance(municipios, str):
        municipios = [municipios]
    data["municipios"] = _dedupe_strs([str(m).strip() for m in municipios if str(m).strip()])

    ufs = data.get("ufs") or []
    if isinstance(ufs, str):
        ufs = [ufs]
    data["ufs"] = _dedupe_strs([str(u).strip().upper()[:2] for u in ufs if str(u).strip()])
    data["ufs"] = [uf for uf in data["ufs"] if uf in UF_SIGLAS]

    for key in ("ano", "semana", "limite"):
        value = data.get(key)
        if value in ("", None):
            data[key] = None if key != "limite" else 10
            continue
        try:
            data[key] = int(value)
        except Exception:
            data[key] = None if key != "limite" else 10

    if data["limite"] is None:
        data["limite"] = 10
    data["limite"] = max(1, min(int(data["limite"]), 10))

    metric = str(data.get("metrica") or "casos").strip().lower()
    data["metrica"] = "incidencia" if metric.startswith("incid") else "casos"

    period = str(data.get("periodo") or "anual").strip().lower()
    data["periodo"] = "semanal" if period.startswith("seman") else "anual"

    if data["semana"] is not None and not (1 <= int(data["semana"]) <= 53):
        data["semana"] = None

    return data


def _build_rule_based_interpretation(question: str) -> dict[str, Any]:
    out = _default_interpretation()
    out["intencao"] = _detect_intent(question)
    out["metrica"] = _detect_metric(question, out["intencao"])
    out["periodo"] = _detect_period(question, out["intencao"])
    out["ano"] = _extract_year(question)
    out["semana"] = _extract_week(question)
    out["limite"] = _extract_limit(question)
    out["ufs"] = _extract_ufs(question)

    municipios: list[str] = []

    if out["intencao"] == "comparacao":
        municipios = _extract_compare_municipios(question)
    else:
        municipio, uf_local = _extract_primary_location(question)
        if municipio:
            municipios.append(municipio)
        if uf_local:
            out["ufs"].append(uf_local)

    out["municipios"] = _dedupe_strs(municipios)
    out["ufs"] = _dedupe_strs(out["ufs"])

    if out["intencao"] == "desconhecida" and "incid" in _norm(question):
        out["intencao"] = "incidencia"
        out["metrica"] = "incidencia"

    return _normalize_interpretation(out)


def _should_try_llm(parsed: dict[str, Any]) -> bool:
    if parsed["intencao"] == "desconhecida":
        return True
    if parsed["intencao"] in {"comparacao"} and len(parsed["municipios"]) < 2:
        return True
    if parsed["intencao"] in {"incidencia", "total", "tendencia", "media_estado", "pico"} and not parsed["municipios"] and not parsed["ufs"]:
        return True
    return False


def _try_llm_interpretation(question: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_S)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = (
        "Você interpreta perguntas epidemiológicas sobre dengue. "
        "Retorne somente JSON válido, sem markdown e sem texto extra. "
        "Campos obrigatórios: intencao, municipios, ufs, ano, semana, metrica, periodo, limite. "
        "Intenções válidas: incidencia, total, ranking, comparacao, tendencia, media_estado, pico, desconhecida."
    )
    user_prompt = (
        f"Pergunta: {question}\n\n"
        "Retorne JSON no formato:\n"
        "{\n"
        '  "intencao": "incidencia | total | ranking | comparacao | tendencia | media_estado | pico | desconhecida",\n'
        '  "municipios": ["Palmas"],\n'
        '  "ufs": ["TO"],\n'
        '  "ano": 2025,\n'
        '  "semana": null,\n'
        '  "metrica": "casos | incidencia",\n'
        '  "periodo": "anual | semanal",\n'
        '  "limite": 10\n'
        "}"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        if isinstance(data, dict):
            return _normalize_interpretation(data)
    except Exception:
        return None
    return None


def interpret_question(pergunta: str) -> dict[str, Any]:
    """
    Interpreta a pergunta em JSON estruturado.

    Estratégia:
    1. Regex/regras simples.
    2. OpenAI somente se necessário, retornando apenas JSON.
    3. Fallback final: resultado por regras.
    """
    base = _build_rule_based_interpretation(pergunta)
    if _should_try_llm(base):
        llm_data = _try_llm_interpretation(pergunta)
        if llm_data:
            merged = _normalize_interpretation(
                {
                    **llm_data,
                    "ano": llm_data.get("ano") if llm_data.get("ano") is not None else base.get("ano"),
                    "semana": llm_data.get("semana") if llm_data.get("semana") is not None else base.get("semana"),
                    "municipios": llm_data.get("municipios") or base.get("municipios"),
                    "ufs": llm_data.get("ufs") or base.get("ufs"),
                    "limite": llm_data.get("limite") or base.get("limite"),
                }
            )
            return merged
    return base
