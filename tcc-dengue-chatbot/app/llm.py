"""Integração com OpenAI (SDK oficial) para geração de respostas (MVP)."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

# Timeout HTTP para chamadas OpenAI (segundos).
OPENAI_TIMEOUT_S = 20.0


def openai_env_status() -> tuple[bool, str]:
    """Indica se variáveis necessárias estão definidas (sem chamar a API)."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or not api_key.strip():
        return False, "OPENAI_API_KEY não definida ou vazia."
    if not model or not model.strip():
        return False, "OPENAI_MODEL não definido."
    return True, f"Modelo configurado: {model}"


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or "",
    timeout=OPENAI_TIMEOUT_S,
)


def _format_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    except Exception:
        return str(obj)


def _sources_note(dados_calculados: Any, contexto_rag: list[dict]) -> str:
    """
    Determina quais fontes citar no final.
    Regra MVP: citar Sinan/Dengue sempre que houver 'casos' em dados/calculos;
    citar IBGE quando houver 'populacao' ou 'incidencia_100k' em dados/conteúdo.
    """

    text = _format_json({"dados_calculados": dados_calculados, "contexto_rag": contexto_rag})
    lower = text.lower()
    has_casos = "casos" in lower or "sinan/dengue" in lower
    has_pop_or_inc = "populacao" in lower or "incidencia_100k" in lower or "ibge" in lower

    sources: list[str] = []
    if has_casos:
        sources.append("Sinan/Dengue")
    if has_pop_or_inc:
        sources.append("IBGE")

    if not sources:
        # fallback mínimo
        sources.append("Sinan/Dengue")
    return "Fontes: " + " e ".join(sources) + "."


def generate_answer(
    pergunta: str,
    dados_calculados: Any,
    contexto_rag: list[dict],
) -> str:
    """
    Gera resposta usando somente DADOS_CALCULADOS e CONTEXTO_RAG.
    Retorna fallback mostrando os dados calculados se a API falhar.

    CONTEXTO_RAG é truncado a no máximo 5 cartas (defesa em profundidade).
    """
    contexto_rag = list(contexto_rag or [])[:5]

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Sem chave -> fallback (não quebrar a aplicação)
    if not api_key or not api_key.strip():
        return (
            "Não foi possível gerar a resposta porque `OPENAI_API_KEY` não está definida.\n\n"
            "Dados calculados:\n"
            + _format_json(dados_calculados)
        )

    system_prompt = (
        "Você é um assistente analítico de saúde pública especializado em dengue.\n"
        "Responda em português brasileiro, de forma conversacional, clara e objetiva — "
        "como um analista explicando resultados para um gestor.\n"
        "Use somente os dados calculados por SQL/Python e o contexto RAG fornecido.\n"
        "Não invente números.\n"
        "Não faça cálculos novos.\n"
        "Não explique fórmulas nem metodologia de incidência; cite apenas os valores já calculados.\n"
        "Se houver ano_assumido em DADOS_CALCULADOS, informe explicitamente que o ano foi assumido "
        "como o mais recente disponível.\n"
        "Em comparações, destaque quem tem maior incidência ou casos e use diferenca_percentual "
        "somente se já estiver presente nos dados.\n"
        "Se faltarem dados, explique o que falta e sugira como reformular a pergunta.\n"
        "Sempre informe fontes: Sinan/Dengue para casos e IBGE para população.\n"
        "Formato sugerido:\n"
        "1. Resposta direta.\n"
        "2. Dados principais (por entidade comparada, se houver).\n"
        "3. Interpretação breve.\n"
        "4. Fontes."
    )

    extra_notes: list[str] = []
    if isinstance(dados_calculados, dict):
        if dados_calculados.get("ano_assumido"):
            extra_notes.append(
                f"- O ano {dados_calculados['ano_assumido']} foi assumido por ser o mais recente disponível."
            )
        if dados_calculados.get("tipo_consulta") == "comparacao":
            extra_notes.append(
                "- Esta é uma pergunta comparativa: organize a resposta por entidade e indique qual ficou acima/abaixo."
            )

    user_prompt = (
        f"PERGUNTA: {pergunta}\n\n"
        f"DADOS_CALCULADOS (somente isto pode ser usado; não inventar):\n{_format_json(dados_calculados)}\n\n"
        f"CONTEXTO_RAG (cartas relevantes; use apenas como suporte):\n{_format_json(contexto_rag)}\n\n"
        "TAREFA:\n"
        "- Responder com base apenas nas seções acima.\n"
        "- Não calcular valores além dos já presentes em DADOS_CALCULADOS.\n"
        "- Não incluir fórmulas matemáticas nem parágrafo sobre metodologia de incidência.\n"
        + ("\n".join(extra_notes) + "\n" if extra_notes else "")
        + "- Se faltarem dados, diga isso claramente e sugira uma pergunta alternativa concreta.\n\n"
        f"{_sources_note(dados_calculados, contexto_rag)}"
    )

    try:
        # Preferir Responses API (quando disponível); fallback para Chat Completions.
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # output_text existe na SDK moderna
            text = getattr(resp, "output_text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            pass

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        return (text or "").strip() or "Resposta vazia retornada pela API."
    except Exception as exc:
        # Timeout ou erro de rede/API: fallback com dados SQL (não quebra a app)
        reason = "tempo limite (20s) excedido" if "timeout" in str(exc).lower() else str(exc)
        return (
            "Não foi possível gerar a resposta via OpenAI neste momento "
            f"({reason}).\n\n"
            "Dados calculados (SQL) para referência:\n"
            + _format_json(dados_calculados)
        )


