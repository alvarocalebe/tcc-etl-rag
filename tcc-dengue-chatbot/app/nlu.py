"""NLU híbrido para interpretação de perguntas epidemiológicas."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
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
    "panorama",
    "desconhecida",
}

EXPLICIT_INTENTS = ALLOWED_INTENTS - {"panorama", "desconhecida"}

_LOCATION_STOPWORDS = frozenset(
    {"como", "qual", "quando", "onde", "quais", "quantos", "porque", "por", "que"}
)


VAGUE_PHRASE_KEYS = (
    "sobre",
    "quero saber",
    "como esta",
    "como está",
    "me fale",
    "me conte",
    "informacoes sobre",
    "informações sobre",
)

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

# Siglas que também são palavras comuns — só aceitar com contexto explícito.
AMBIGUOUS_UF_SIGLAS = {"AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"}

UF_FUZZY_MIN_SCORE = 0.82

UF_TO_STATE_LABEL = {
    uf: name.title()
    for name, uf in STATE_NAME_TO_UF.items()
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
        "entidades": [],
        "ano": None,
        "semana": None,
        "metrica": "casos",
        "periodo": "anual",
        "limite": 10,
        "gerar_grafico": False,
    }


def wants_chart(question: str) -> bool:
    """Pergunta pede visualização/gráfico."""
    text = _norm(question)
    return any(
        k in text
        for k in (
            "grafico",
            "gráfico",
            "chart",
            "visualiz",
            "plotar",
            "plote",
            "mostre um grafico",
            "mostre um gráfico",
            "gerar grafico",
            "gerar gráfico",
            "gere um grafico",
            "gere um gráfico",
        )
    )


def wants_comparative_chart(question: str) -> bool:
    text = _norm(question)
    if not wants_chart(question):
        return False
    return any(
        k in text
        for k in (
            "comparativ",
            "comparando",
            "compare",
            "comparar",
            "comparacao",
            "versus",
            " vs ",
            "entre ",
        )
    ) or "compar" in text


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


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _fuzzy_match_state_name(name: str) -> tuple[str | None, float]:
    norm = _norm(name)
    if not norm or len(norm) < 3:
        return None, 0.0

    if norm in STATE_NAME_TO_UF:
        return STATE_NAME_TO_UF[norm], 1.0

    best_uf: str | None = None
    best_score = 0.0
    for state_name, uf in STATE_NAME_TO_UF.items():
        score = _similarity(norm, state_name)
        if score > best_score:
            best_score = score
            best_uf = uf

    min_score = UF_FUZZY_MIN_SCORE if len(norm) >= 6 else 0.9
    if best_score >= min_score:
        return best_uf, best_score
    return None, best_score


def resolve_uf(term: str) -> str | None:
    """
    Resolve sigla ou nome de estado (com tolerância a maiúsculas/minúsculas e typos leves).
    """
    raw = str(term or "").strip()
    if not raw:
        return None

    sigla = raw.upper()[:2]
    if len(sigla) == 2 and sigla in UF_SIGLAS and re.fullmatch(r"[A-Za-z]{2}", raw.strip()):
        return sigla

    uf, _ = _fuzzy_match_state_name(raw)
    return uf


def is_likely_state_name(term: str) -> bool:
    norm = _norm(term)
    if not norm:
        return False
    if norm in STATE_NAME_TO_UF:
        return True
    uf, score = _fuzzy_match_state_name(term)
    return uf is not None and score >= UF_FUZZY_MIN_SCORE and len(norm) >= 5


def normalize_ufs(ufs: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw in ufs or []:
        uf = resolve_uf(raw)
        if uf:
            resolved.append(uf)
        else:
            sigla = str(raw).strip().upper()[:2]
            if sigla in UF_SIGLAS:
                resolved.append(sigla)
    return _dedupe_strs(resolved)


def _extract_ufs_from_siglas(question: str) -> list[str]:
    text = str(question or "")
    text_norm = _norm(text)
    ufs: list[str] = []

    for pattern in (
        r"\b(?:do|da|de|no|na|em|uf|estado)\s+([A-Za-z]{2})\b",
        r"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]+?)\s+([A-Za-z]{2})\s+(?:em|no|na)\s+(?:19\d{2}|20\d{2})\b",
        r"\b([A-Za-z]{2})\s+(?:em|no|na)\s+(?:19\d{2}|20\d{2})\b",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(match.lastindex or 1).upper()
            if candidate in UF_SIGLAS:
                ufs.append(candidate)

    for uf in sorted(UF_SIGLAS):
        if uf in AMBIGUOUS_UF_SIGLAS:
            continue
        if re.search(rf"\b{uf}\b", text.upper()):
            ufs.append(uf)

    for uf in sorted(AMBIGUOUS_UF_SIGLAS):
        if re.search(rf"\b(?:do|da|de|no|na|em|uf|estado)\s+{uf}\b", text, flags=re.IGNORECASE):
            ufs.append(uf)
        if re.search(rf"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]+?)\s+{uf}\b", text, flags=re.IGNORECASE):
            ufs.append(uf)

    return _dedupe_strs(ufs)


def _extract_ufs_from_state_names(question: str) -> list[str]:
    text_norm = _norm(question)
    ufs: list[str] = []

    for state_name, uf in sorted(STATE_NAME_TO_UF.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(state_name)}\b", text_norm):
            ufs.append(uf)

    for match in re.finditer(
        r"\b(?:do|da|de|no|na|em|estado|uf)\s+([a-zà-ÿ\s]{2,30}?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
        text_norm,
    ):
        phrase = match.group(1).strip()
        words = phrase.split()
        for size in range(min(4, len(words)), 0, -1):
            candidate = " ".join(words[:size])
            uf, score = _fuzzy_match_state_name(candidate)
            if uf and score >= UF_FUZZY_MIN_SCORE:
                ufs.append(uf)
                break

    tokens = re.findall(r"[a-zà-ÿ]{4,}", text_norm)
    for token in tokens:
        if token in STATE_NAME_TO_UF:
            continue
        uf, score = _fuzzy_match_state_name(token)
        if uf and score >= 0.9 and len(token) >= 6:
            ufs.append(uf)

    return _dedupe_strs(ufs)


def _extract_ufs(question: str) -> list[str]:
    ufs = _extract_ufs_from_state_names(question)
    ufs.extend(_extract_ufs_from_siglas(question))
    return normalize_ufs(ufs)


def _split_location_and_uf(location: str) -> tuple[str, str | None]:
    raw = str(location or "").strip(" .,:;?!")
    if not raw:
        return "", None

    for sep in ("-", "/"):
        if sep in raw:
            left, right = raw.rsplit(sep, 1)
            sigla = right.strip().upper()[:2]
            if len(sigla) == 2 and sigla in UF_SIGLAS and re.fullmatch(
                r"[A-Za-z]{2}", right.strip()
            ):
                nome = left.strip(" ,-/")
                if nome:
                    return nome, sigla

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

    parts = raw.split()
    if len(parts) >= 2:
        tail = parts[-1]
        uf = resolve_uf(tail)
        if uf and re.fullmatch(r"[A-Za-z]{2}", tail.strip()):
            return " ".join(parts[:-1]).strip(), uf

    uf, score = _fuzzy_match_state_name(raw)
    if uf and score >= UF_FUZZY_MIN_SCORE and len(norm_raw) >= 5:
        return "", uf
    return raw, None


_TEMPORAL_SUFFIX_PATTERNS = (
    r"\s+durante\s+o\s+ano\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+no\s+ano\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+ao\s+longo\s+do\s+ano\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+ao\s+longo\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+no\s+periodo\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+no\s+período\s+de\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+durante\s+(?:o\s+)?ano\s+(?:de\s+)?(?:19\d{2}|20\d{2})\s*$",
    r"\s+no\s+ano\s+(?:de\s+)?(?:19\d{2}|20\d{2})\s*$",
    r"\s+em\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+no\s+(?:19\d{2}|20\d{2})\s*$",
    r"\s+ao\s+longo\s+de\s+(?:19\d{2}|20\d{2})\s*$",
)

_MUNICIPIO_NOISE_RE = re.compile(
    r"\b(?:dengue|incidencia|incidência|casos|qual|foi|foram|quantos|total|"
    r"durante|ano|anos|periodo|período|semana|semanas|epidemiologica|"
    r"epidemiológica|ultimas|últimas|municipio|município|cidade|estado|"
    r"regiao|região|sobre|dados|epidemia|taxa|media|média|notificados|"
    r"confirmados|habitantes|situacao|situação|informar|gostaria|poderia)\b",
    re.IGNORECASE,
)


def _strip_temporal_suffix(location: str) -> str:
    """Remove sufixos temporais que às vezes entram no nome do município."""
    value = str(location or "").strip(" .,:;?!")
    if not value:
        return value
    changed = True
    while changed:
        changed = False
        for pattern in _TEMPORAL_SUFFIX_PATTERNS:
            new_value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" .,:;?!")
            if new_value != value:
                value = new_value
                changed = True
    return value


def sanitize_municipio_name(name: str) -> str:
    """Normaliza nome de município removendo ruído temporal e epidemiológico."""
    value = _clean_location_phrase(_strip_temporal_suffix(str(name or "").strip()))
    if not value:
        return ""
    while True:
        stripped = re.sub(
            r"^(?:o|a|os|as|de|da|do|das|dos|em|no|na|nos|nas)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip(" .,:;?!")
        if stripped == value:
            break
        value = stripped
    value = re.sub(r"\b(?:19|20)\d{2}\b", "", value)
    value = _MUNICIPIO_NOISE_RE.sub(" ", value)
    value = re.sub(r"\s{2,}", " ", value).strip(" .,:;?!")
    return value


def _is_valid_municipio_candidate(name: str | None) -> bool:
    if not name:
        return False
    norm = _norm(str(name))
    if norm in _LOCATION_STOPWORDS:
        return False
    return not is_suspect_municipio_name(str(name))


def is_suspect_municipio_name(name: str) -> bool:
    """Detecta nomes de município provavelmente contaminados pelo restante da pergunta."""
    raw = str(name or "").strip()
    if len(raw) < 2:
        return True
    norm = _norm(raw)
    if len(norm) > 45:
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", raw):
        return True
    markers = (
        "durante",
        " incid",
        " dengue",
        " quantos",
        " qual foi",
        " ano de",
        " no ano",
        " periodo",
        " período",
        " notificad",
        " confirmad",
        " habitante",
        " epidemiolog",
        " ao longo",
    )
    if any(marker in norm for marker in markers):
        return True
    words = [w for w in norm.split() if w not in {"de", "da", "do", "em", "no", "na", "o", "a"}]
    return len(words) > 5


def _clean_location_phrase(location: str) -> str:
    value = str(location or "").strip(" .,:;?!")
    value = re.sub(
        r"\b(?:de dengue|nas ultimas semanas|nas últimas semanas|ultimas semanas|últimas semanas)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = _strip_temporal_suffix(value)
    value = re.sub(r"\s{2,}", " ", value).strip(" .,:;?!")
    return value


def _strip_metric_prefix(phrase: str) -> str:
    value = str(phrase or "").strip(" .,:;?!")
    value = re.sub(
        r"^(?:a\s+)?(?:media|média|incidencia|incidência)\s+(?:do|da|de)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" .,:;?!")


def _strip_state_prefix(phrase: str) -> str:
    """Remove prefixos como 'estado do Paraná' / 'o estado de São Paulo'."""
    value = str(phrase or "").strip(" .,:;?!")
    if not value:
        return value
    m = re.match(
        r"^(?:o\s+|a\s+)?(?:estado|uf)\s+(?:do|da|de)\s+(.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(" .,:;?!")
    m = re.match(
        r"^(?:media|média)\s+(?:do|da|de)\s+(?:estado|uf)\s+(?:do|da|de)\s+(.+)$",
        value,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip(" .,:;?!")
    return value


def _uf_label(uf: str) -> str:
    sigla = str(uf or "").strip().upper()[:2]
    nome = UF_TO_STATE_LABEL.get(sigla, sigla)
    return f"{nome} ({sigla})"


def _parse_location_entity(raw: str) -> dict[str, Any] | None:
    cleaned = _strip_state_prefix(_strip_metric_prefix(_clean_location_phrase(raw)))
    if not cleaned:
        return None

    if is_likely_state_name(cleaned):
        uf = resolve_uf(cleaned)
        if uf:
            return {"tipo": "uf", "uf": uf, "nome": None, "label": _uf_label(uf)}

    nome, uf_hint = _split_location_and_uf(cleaned)

    if not nome and uf_hint:
        return {"tipo": "uf", "uf": uf_hint, "nome": None, "label": _uf_label(uf_hint)}

    if not nome:
        uf = resolve_uf(cleaned)
        if uf:
            return {"tipo": "uf", "uf": uf, "nome": None, "label": _uf_label(uf)}
        return None

    if is_likely_state_name(nome) and not uf_hint:
        uf = resolve_uf(nome)
        if uf:
            return {"tipo": "uf", "uf": uf, "nome": None, "label": _uf_label(uf)}

    label = f"{nome}/{uf_hint}" if uf_hint else nome
    return {"tipo": "municipio", "nome": nome, "uf": uf_hint, "label": label}


def _extract_comparison_entities(question: str) -> list[dict[str, Any]]:
    text = str(question or "").strip()
    if not text:
        return []

    compare_patterns = (
        r"\b(?:compare|comparar|comparacao|comparação)\s+(.+?)\s+(?:com|com a|versus|vs\.?|e)\s+(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
        r"(?:media|média)\s+(?:do|da|de)\s+(.+?)\s+(?:com|com a|versus|vs\.?|e)\s+(?:a\s+)?(?:media|média)\s+(?:do|da|de)\s+(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
        r"\b(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
        r"\b(?:grafico|gráfico)\s+comparativ[oa]?\s+entre\s+(.+?)\s+e\s+(?:o\s+|a\s+)?(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
        r"\bentre\s+(.+?)\s+e\s+(?:o\s+|a\s+)?(.+?)(?:\s+em\s+(?:19\d{2}|20\d{2})|\?|$|,|\.)",
    )

    for pattern in compare_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        left = _parse_location_entity(match.group(1))
        right = _parse_location_entity(match.group(2))
        entities = [entity for entity in (left, right) if entity]
        if len(entities) >= 2:
            return entities

    legacy = _extract_compare_municipios(question)
    entities: list[dict[str, Any]] = []
    for name in legacy:
        entity = _parse_location_entity(name)
        if entity:
            entities.append(entity)
    return entities


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


def _location_from_phrase(phrase: str) -> tuple[str | None, str | None]:
    location = _clean_location_phrase(phrase)
    if not location:
        return None, None
    if re.fullmatch(r"\d{4}", location):
        return None, None
    if _norm(location) in {"a dengue", "dengue", "dengue no brasil"}:
        return None, None
    if location.upper() in UF_SIGLAS or _norm(location) in STATE_NAME_TO_UF:
        uf_only = resolve_uf(location)
        return None, uf_only
    municipio, uf = _split_location_and_uf(location)
    municipio = _clean_location_phrase(municipio)
    if municipio:
        return municipio, uf
    if not municipio and uf:
        return None, uf
    return None, None


def _looks_like_analytical_question(question: str) -> bool:
    text = _norm(question)
    markers = (
        "esta acima",
        "está acima",
        "esta abaixo",
        "está abaixo",
        "media do",
        "média do",
        "compare",
        "comparar",
        "versus",
        "quantos",
        "incidencia",
        "incidência",
        "ranking",
        "tendencia",
        "tendência",
        "pico",
    )
    return any(m in text for m in markers)


def _extract_bare_location(question: str) -> tuple[str | None, str | None]:
    """
    Extrai localidade de frases curtas ou vagas (ex.: 'palmas tocantins', 'Serra Talhada').
    """
    text = str(question or "").strip()
    if not text or _looks_like_analytical_question(text):
        return None, None

    vague_patterns = (
        r"\b(?:quero saber|gostaria de saber|me fale|me conte)(?:\s+sobre)?\s+(.+?)(?:\?|$)",
        r"\bsobre\s+(.+?)(?:\?|$)",
        r"\bcomo\s+esta\s+(?:a\s+)?(?:dengue\s+)?(?:em|no|na)\s+(.+?)(?:\?|$)",
        r"\bcomo\s+está\s+(?:a\s+)?(?:dengue\s+)?(?:em|no|na)\s+(.+?)(?:\?|$)",
    )
    for pattern in vague_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            municipio, uf = _location_from_phrase(m.group(1))
            if municipio or uf:
                return municipio, uf

    stripped = re.sub(
        r"^(?:quero saber|gostaria de saber|me fale|me conte|informacoes sobre|informações sobre)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .,:;?!")
    stripped = re.sub(r"^(?:sobre|de|da|do)\s+", "", stripped, flags=re.IGNORECASE).strip(" .,:;?!")
    if stripped and len(stripped) <= 80:
        municipio, uf = _location_from_phrase(stripped)
        if municipio or uf:
            return municipio, uf

    return None, None


def _extract_location_from_elaborate_phrase(question: str) -> tuple[str | None, str | None]:
    """Padrões para perguntas longas (banca, formulários elaborados)."""
    text = str(question or "").strip()
    if not text:
        return None, None

    patterns = (
        r"\b(?:na|cidade|municipio|município)\s+(?:de|do|da)\s+(.+?)(?:\s+(?:durante|no\s+ano|ao\s+longo|em\s+(?:19|20)\d{2})|\?|$)",
        r"\b(?:para|em)\s+(?:o\s+|a\s+)?(?:municipio|município|cidade)\s+(?:de|do|da)\s+(.+?)(?:\?|$)",
        r"\b(?:dados|casos|incidencia|incidência|situacao|situação|taxa)\s+(?:de|da|do)?\s*dengue\s+(?:em|no|na)\s+(.+?)(?:\s+(?:durante|no\s+ano)|\?|$)",
        r"\b(?:notificados|confirmados|registrados)\s+(?:em|no|na)\s+(.+?)(?:\s+(?:no\s+ano|em\s+(?:19|20)\d{2})|\?|$)",
        r"\b(?:quantos|qual)\s+.+?\s+(?:em|no|na)\s+(.+?)(?:\s+(?:durante|no\s+ano|ao\s+longo)|\?|$)",
        r"\b(?:em|no|na)\s+(.+?)\s+[-/]\s*([A-Za-z]{2})\b",
    )

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        chunk = m.group(1)
        municipio, uf = _location_from_phrase(chunk)
        if m.lastindex and m.lastindex >= 2:
            sigla = str(m.group(2) or "").strip().upper()[:2]
            if sigla in UF_SIGLAS:
                uf = uf or sigla
        if municipio or uf:
            return municipio, uf
    return None, None


def _extract_all_location_hints(question: str) -> tuple[str | None, str | None]:
    """Tenta várias heurísticas até obter localidade plausível."""
    text = str(question or "").strip()
    extractors = (
        _extract_location_from_elaborate_phrase,
        _extract_primary_location,
    )
    for extractor in extractors:
        municipio, uf = extractor(text)
        if municipio:
            municipio = sanitize_municipio_name(municipio)
        if municipio and not is_suspect_municipio_name(municipio):
            return municipio, uf
        if uf and not municipio:
            return None, uf

    bare_mun, bare_uf = _extract_bare_location(text)
    if bare_mun:
        bare_mun = sanitize_municipio_name(bare_mun)
    if bare_mun and not is_suspect_municipio_name(bare_mun):
        return bare_mun, bare_uf
    if bare_uf:
        return None, bare_uf
    return None, None


def _extract_primary_location(question: str) -> tuple[str | None, str | None]:
    text = str(question or "").strip()

    patterns = [
        r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]{1,60}?)\s+esta\b",
        r"^\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\- ]{1,60}?)\s+está\b",
        r"\b(?:em|no|na)\s+(.+?)\s+em\s+(?:19\d{2}|20\d{2})\b",
        r"\b(?:em|no|na)\s+(.+?)\s+(?:durante\s+)?(?:o\s+)?ano\s+(?:de\s+)?(?:19\d{2}|20\d{2})\b",
        r"\b(?:em|no|na)\s+(.+?)\s+durante\s+(?:19\d{2}|20\d{2})\b",
        r"\b(?:em|no|na)\s+(.+?)\s+ao\s+longo\s+(?:de\s+)?(?:19\d{2}|20\d{2})\b",
        r"\b(?:em|no|na)\s+(.+?)(?:\s+nas?\s+ultimas?\s+semanas|\s+nas?\s+últimas?\s+semanas|\?|$)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        municipio, uf = _location_from_phrase(m.group(1))
        if municipio and not _is_valid_municipio_candidate(municipio):
            municipio = None
        if municipio or uf:
            return municipio, uf

    elaborate_mun, elaborate_uf = _extract_location_from_elaborate_phrase(text)
    if elaborate_mun or elaborate_uf:
        return elaborate_mun, elaborate_uf

    bare_mun, bare_uf = _extract_bare_location(text)
    if bare_mun or bare_uf:
        return bare_mun, bare_uf
    return None, None


def _is_comparison_question(question: str) -> bool:
    text = _norm(question)
    if re.search(r"\b(?:compare|comparar|comparacao|comparacao|versus|vs\.?)\b", text):
        return True
    if re.search(r"(?:media|média).*\b(?:com|com a|versus|vs\.?| e )\b", question, flags=re.IGNORECASE):
        return True
    if wants_comparative_chart(question):
        return True
    if wants_chart(question) and re.search(
        r"\b(?:entre|com|e|versus|vs\.?)\b", text
    ):
        return True
    return False


def _detect_intent(question: str) -> str:
    text = _norm(question)

    if any(key in text for key in ["semana com mais casos", "pico epidemiologico", "pico epidemiológico", "pico de casos"]):
        return "pico"
    if _is_comparison_question(question):
        return "comparacao"
    if any(key in text for key in ["acima da media", "abaixo da media", "acima da média", "abaixo da média", "media estadual", "média estadual"]):
        return "media_estado"
    if any(key in text for key in ["media do", "média do"]) and not _is_comparison_question(question):
        return "media_estado"
    if any(key in text for key in ["aumentando", "tendencia", "tendência", "ultimas semanas", "últimas semanas"]):
        return "tendencia"
    if any(key in text for key in ["ranking", "maior incidencia", "maior incidência", "maiores incidencias", "maiores incidências", "maior incid", "mais casos", "quais cidades", "quais municipios", "quais municípios"]):
        return "ranking"
    if any(key in text for key in ["incidencia", "incidência", "por 100 mil", "100 mil"]):
        return "incidencia"
    if any(key in text for key in ["quantos casos", "total de casos", "casos de dengue", "quantos"]):
        return "total"
    if any(key in text for key in VAGUE_PHRASE_KEYS):
        return "desconhecida"
    return "desconhecida"


def has_location_hint(interpretacao: dict[str, Any]) -> bool:
    return bool(interpretacao.get("municipios") or interpretacao.get("ufs"))


def infer_intent_when_vague(interpretacao: dict[str, Any], pergunta: str = "") -> dict[str, Any]:
    """
    Se há localidade mas intenção indefinida, assume panorama (resumo epidemiológico).
    """
    data = dict(interpretacao or {})
    intent = str(data.get("intencao") or "desconhecida")
    if intent in EXPLICIT_INTENTS:
        return data
    if not has_location_hint(data):
        return data
    data["intencao"] = "panorama"
    if not data.get("metrica"):
        data["metrica"] = "casos"
    return data


def _detect_metric(question: str, intent: str) -> str:
    text = _norm(question)
    if intent in {"incidencia", "media_estado"}:
        return "incidencia"
    if intent == "comparacao":
        if any(key in text for key in ["media", "média", "incid"]):
            return "incidencia"
        return "casos"
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
    data["municipios"] = _dedupe_strs(
        [sanitize_municipio_name(str(m)) for m in municipios if str(m).strip()]
    )
    data["municipios"] = [m for m in data["municipios"] if m]

    ufs = data.get("ufs") or []
    if isinstance(ufs, str):
        ufs = [ufs]
    data["ufs"] = normalize_ufs([str(u).strip() for u in ufs if str(u).strip()])

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

    data["gerar_grafico"] = bool(data.get("gerar_grafico"))

    if data["semana"] is not None and not (1 <= int(data["semana"]) <= 53):
        data["semana"] = None

    entidades = data.get("entidades") or []
    if isinstance(entidades, dict):
        entidades = [entidades]
    normalized_entities: list[dict[str, Any]] = []
    for raw_entity in entidades:
        if not isinstance(raw_entity, dict):
            continue
        tipo = str(raw_entity.get("tipo") or "").strip().lower()
        if tipo not in {"uf", "municipio"}:
            continue
        entity: dict[str, Any] = {"tipo": tipo}
        if tipo == "uf":
            uf = resolve_uf(str(raw_entity.get("uf") or raw_entity.get("nome") or ""))
            if not uf:
                continue
            entity["uf"] = uf
            entity["nome"] = None
            entity["label"] = str(raw_entity.get("label") or _uf_label(uf))
        else:
            nome = _clean_location_phrase(
                _strip_temporal_suffix(
                    str(raw_entity.get("nome") or raw_entity.get("municipio") or "").strip()
                )
            )
            if len(nome) < 2:
                continue
            uf_hint = resolve_uf(str(raw_entity.get("uf") or "")) if raw_entity.get("uf") else None
            entity["nome"] = nome
            entity["uf"] = uf_hint
            entity["label"] = str(raw_entity.get("label") or (f"{nome}/{uf_hint}" if uf_hint else nome))
        normalized_entities.append(entity)
    data["entidades"] = normalized_entities

    municipios = list(data.get("municipios") or [])
    ufs = list(data.get("ufs") or [])
    for entity in normalized_entities:
        if entity["tipo"] == "uf" and entity.get("uf"):
            ufs.append(str(entity["uf"]))
        elif entity["tipo"] == "municipio" and entity.get("nome"):
            municipios.append(str(entity["nome"]))
            if entity.get("uf"):
                ufs.append(str(entity["uf"]))
    data["municipios"] = _dedupe_strs(municipios)
    data["ufs"] = normalize_ufs([str(u) for u in ufs if str(u).strip()])

    return data


def repair_interpretation(question: str, data: dict[str, Any]) -> dict[str, Any]:
    """
    Corrige interpretações com município contaminado ou ausente (perguntas longas).
    """
    out = relocate_state_names(dict(data or {}))
    pergunta = str(question or "").strip()

    municipios = [
        sanitize_municipio_name(str(m))
        for m in list(out.get("municipios") or [])
        if str(m).strip()
    ]
    municipios = [m for m in municipios if m and not is_likely_state_name(m)]
    ufs = list(out.get("ufs") or [])

    needs_repair = not municipios or any(
        is_suspect_municipio_name(str(m)) for m in list(out.get("municipios") or [])
    )

    if needs_repair:
        mun, uf = _extract_all_location_hints(pergunta)
        if mun:
            municipios = _dedupe_strs([mun])
        if uf:
            ufs.append(uf)

    if out.get("intencao") == "comparacao":
        entidades = list(out.get("entidades") or [])
        repaired_entities: list[dict[str, Any]] = []
        for entity in entidades:
            if not isinstance(entity, dict):
                continue
            entity = dict(entity)
            if entity.get("tipo") == "municipio" and entity.get("nome"):
                nome = sanitize_municipio_name(str(entity["nome"]))
                if is_suspect_municipio_name(nome):
                    continue
                entity["nome"] = nome
                entity["label"] = str(entity.get("label") or nome)
            repaired_entities.append(entity)
        if len(repaired_entities) < 2 and _is_comparison_question(pergunta):
            repaired_entities = _extract_comparison_entities(pergunta)
        out["entidades"] = repaired_entities

    if not out.get("ano"):
        ano = _extract_year(pergunta)
        if ano is not None:
            out["ano"] = ano

    out["municipios"] = _dedupe_strs(municipios)
    out["ufs"] = _dedupe_strs(ufs)
    return _normalize_interpretation(out)


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
        out["entidades"] = _extract_comparison_entities(question)
        municipios = [
            str(entity["nome"])
            for entity in out["entidades"]
            if entity.get("tipo") == "municipio" and entity.get("nome")
        ]
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

    out["gerar_grafico"] = wants_comparative_chart(question) or (
        wants_chart(question) and out["intencao"] == "comparacao"
    )

    return _normalize_interpretation(out)


def relocate_state_names(interpretacao: dict[str, Any]) -> dict[str, Any]:
    """
    Move termos que parecem nome de estado da lista de municípios para UFs.
    """
    data = dict(interpretacao or {})
    municipios = list(data.get("municipios") or [])
    ufs = list(data.get("ufs") or [])
    kept: list[str] = []

    for term in municipios:
        if is_likely_state_name(term):
            uf = resolve_uf(term)
            if uf:
                ufs.append(uf)
                continue
        kept.append(term)

    data["municipios"] = _dedupe_strs(kept)
    data["ufs"] = normalize_ufs(ufs)
    return data


def count_valid_entities(parsed: dict[str, Any]) -> int:
    entidades = list(parsed.get("entidades") or [])
    if entidades:
        return len(entidades)
    municipios = list(parsed.get("municipios") or [])
    ufs = list(parsed.get("ufs") or [])
    if parsed.get("intencao") == "comparacao":
        return len(municipios) + len(ufs)
    return len(municipios) + len(ufs)


def _is_vague_question(question: str) -> bool:
    text = _norm(question)
    if any(key in text for key in VAGUE_PHRASE_KEYS):
        return True
    tokens = [t for t in re.findall(r"[a-zà-ÿ]{2,}", text) if t not in {"de", "da", "do", "em", "no", "na", "a", "o"}]
    return len(tokens) <= 6


def _should_try_llm(parsed: dict[str, Any]) -> bool:
    pergunta = str(parsed.get("_pergunta_original") or "")
    if parsed["intencao"] == "desconhecida":
        return True
    if parsed["intencao"] == "comparacao" and count_valid_entities(parsed) < 2:
        return True
    if parsed["intencao"] == "media_estado" and not parsed["municipios"]:
        return True
    if parsed["intencao"] in {"incidencia", "total", "tendencia", "pico"} and not parsed["municipios"] and not parsed["ufs"]:
        return True
    if _is_comparison_question(pergunta) and count_valid_entities(parsed) < 2:
        return True
    if _is_vague_question(pergunta) and not has_location_hint(parsed):
        return True
    if parsed["intencao"] == "desconhecida" and _is_vague_question(pergunta):
        return True
    if any(is_suspect_municipio_name(str(m)) for m in parsed.get("municipios") or []):
        return True
    return False


def _try_llm_interpretation(question: str) -> dict[str, Any] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_S)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = (
        "Você interpreta perguntas epidemiológicas sobre dengue no Brasil. "
        "Retorne somente JSON válido, sem markdown e sem texto extra. "
        "Campos obrigatórios: intencao, municipios, ufs, entidades, ano, semana, metrica, periodo, limite. "
        "Intenções válidas: incidencia, total, ranking, comparacao, tendencia, media_estado, pico, panorama, desconhecida. "
        "Use panorama quando a pergunta for vaga mas houver município ou UF (ex.: 'quero saber sobre X', 'palmas tocantins'). "
        "Nunca use desconhecida se municipios ou ufs estiverem preenchidos — use panorama. "
        "Use gerar_grafico=true quando o usuário pedir gráfico comparativo ou visualização comparando localidades. "
        "Use entidades para comparações mistas entre UF e município. "
        "Cada entidade deve ter: tipo (uf|municipio), uf, nome, label."
    )
    user_prompt = (
        f"Pergunta: {question}\n\n"
        "Exemplos:\n"
        '1) "compare a média do parana com a média de palmas to" -> '
        '{"intencao":"comparacao","entidades":[{"tipo":"uf","uf":"PR","nome":null,"label":"Parana (UF)"},'
        '{"tipo":"municipio","nome":"Palmas","uf":"TO","label":"Palmas/TO"}],"metrica":"incidencia"}\n'
        '2) "compare sao paulo e rio de janeiro em 2025" -> '
        '{"intencao":"comparacao","entidades":[{"tipo":"uf","uf":"SP","nome":null,"label":"Sao Paulo (UF)"},'
        '{"tipo":"uf","uf":"RJ","nome":null,"label":"Rio de Janeiro (UF)"}],"ano":2025,"metrica":"incidencia"}\n'
        '3) "Palmas está acima da média do Tocantins em 2025" -> '
        '{"intencao":"media_estado","municipios":["Palmas"],"ufs":["TO"],"ano":2025,"metrica":"incidencia"}\n'
        '4) "qual a incidencia em araguaina" -> '
        '{"intencao":"incidencia","municipios":["Araguaína"],"metrica":"incidencia"}\n'
        '5) "Quero saber sobre Serra Talhada" -> '
        '{"intencao":"panorama","municipios":["Serra Talhada"],"metrica":"casos"}\n'
        '6) "palmas tocantins" -> '
        '{"intencao":"panorama","municipios":["Palmas"],"ufs":["TO"],"metrica":"casos"}\n'
        '7) "como está a dengue em Araguaína" -> '
        '{"intencao":"panorama","municipios":["Araguaína"],"metrica":"casos"}\n'
        '8) "gere um gráfico comparativo entre Palmas TO e o Paraná em 2025" -> '
        '{"intencao":"comparacao","gerar_grafico":true,"entidades":[{"tipo":"municipio","nome":"Palmas","uf":"TO","label":"Palmas/TO"},'
        '{"tipo":"uf","uf":"PR","nome":null,"label":"Parana (UF)"}],"ano":2025,"metrica":"casos"}\n'
        '9) "qual foi a incidencia da dengue em palmas durante o ano de 2025" -> '
        '{"intencao":"incidencia","municipios":["Palmas"],"ufs":["TO"],"ano":2025,"metrica":"incidencia"}\n'
        'Nunca inclua ano, "durante", "dengue" ou "incidencia" dentro de municipios.\n\n'
        "Retorne JSON no formato:\n"
        "{\n"
        '  "intencao": "incidencia | total | ranking | comparacao | tendencia | media_estado | pico | panorama | desconhecida",\n'
        '  "gerar_grafico": false,\n'
        '  "municipios": ["Palmas"],\n'
        '  "ufs": ["TO"],\n'
        '  "entidades": [{"tipo":"uf","uf":"PR","nome":null,"label":"Parana (UF)"}],\n'
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
    base = relocate_state_names(_build_rule_based_interpretation(pergunta))
    base["_pergunta_original"] = pergunta
    if _should_try_llm(base):
        llm_data = _try_llm_interpretation(pergunta)
        if llm_data:
            merged = relocate_state_names(
                _normalize_interpretation(
                    {
                        **llm_data,
                        "ano": llm_data.get("ano") if llm_data.get("ano") is not None else base.get("ano"),
                        "semana": llm_data.get("semana") if llm_data.get("semana") is not None else base.get("semana"),
                        "municipios": llm_data.get("municipios") or base.get("municipios"),
                        "ufs": llm_data.get("ufs") or base.get("ufs"),
                        "entidades": llm_data.get("entidades") or base.get("entidades"),
                        "limite": llm_data.get("limite") or base.get("limite"),
                    }
                )
            )
            merged.pop("_pergunta_original", None)
            return repair_interpretation(pergunta, infer_intent_when_vague(merged, pergunta))
    base.pop("_pergunta_original", None)
    return repair_interpretation(pergunta, infer_intent_when_vague(base, pergunta))
