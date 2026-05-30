"""Gráficos Plotly para o chatbot (comparações e séries)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go


def wants_comparative_chart(pergunta: str, interpretacao: dict[str, Any] | None = None) -> bool:
    """Detecta pedido explícito de gráfico comparativo na pergunta ou interpretação."""
    if interpretacao and interpretacao.get("gerar_grafico"):
        return True
    text = str(pergunta or "").lower()
    text = text.replace("gráfico", "grafico")
    if "grafico" not in text and "chart" not in text and "visualiz" not in text:
        return False
    comparativo_keys = (
        "comparativ",
        "comparando",
        "compare",
        "comparar",
        "comparacao",
        "versus",
        " vs ",
        " vs.",
        "entre ",
    )
    return any(k in text for k in comparativo_keys) or "compar" in text


def prepare_comparacao_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza DataFrame de comparação com coluna `entidade_label` para o eixo X.
    Suporta saída de comparar_entidades e comparar_municipios.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "entidade_label" not in out.columns or out["entidade_label"].isna().all():
        labels: list[str] = []
        for _, row in out.iterrows():
            if pd.notna(row.get("entidade_label")) and str(row.get("entidade_label")).strip():
                labels.append(str(row["entidade_label"]).strip())
                continue
            nome = row.get("nome_municipio")
            uf = row.get("uf_sigla")
            tipo = row.get("tipo_entidade")
            if pd.notna(nome) and str(nome).strip():
                label = str(nome).strip()
                if pd.notna(uf) and str(uf).strip():
                    label = f"{label}/{uf}"
            elif pd.notna(uf) and str(uf).strip():
                label = str(uf).strip().upper()
            else:
                label = "Entidade"
            if tipo == "uf" and pd.notna(uf):
                label = f"{label} (UF)" if "(UF)" not in label else label
            labels.append(label)
        out["entidade_label"] = labels

    return out.drop_duplicates(subset=["entidade_label"], keep="first").reset_index(drop=True)


def comparacao_chartable(df: pd.DataFrame) -> bool:
    prepared = prepare_comparacao_df(df)
    if prepared.empty or len(prepared) < 2:
        return False
    has_casos = "total_casos" in prepared.columns and prepared["total_casos"].notna().any()
    has_inc = "incidencia_100k" in prepared.columns and prepared["incidencia_100k"].notna().any()
    return has_casos or has_inc


def build_comparacao_figure(
    df: pd.DataFrame,
    *,
    metrica: str = "casos",
    titulo: str | None = None,
) -> Any:
    """
    Gráfico de barras agrupadas: casos e incidência por entidade (município, UF ou misto).
    """
    prepared = prepare_comparacao_df(df)
    if not comparacao_chartable(prepared):
        return None

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = prepared["entidade_label"].astype(str).tolist()
    casos = pd.to_numeric(prepared.get("total_casos"), errors="coerce").fillna(0)
    incid = pd.to_numeric(prepared.get("incidencia_100k"), errors="coerce")

    prefer_inc = str(metrica or "casos").lower().startswith("incid")
    has_inc_values = incid.notna().any() and float(incid.fillna(0).sum()) > 0
    has_casos_values = float(casos.sum()) > 0

    if has_inc_values and (prefer_inc or not has_casos_values):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=incid,
                name="Incidência (/100 mil)",
                marker_color="#2563eb",
                text=[f"{v:.1f}" if pd.notna(v) else "—" for v in incid],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=titulo or "Comparação de incidência",
            xaxis_title="Localidade",
            yaxis_title="Incidência por 100 mil habitantes",
            height=460,
            bargap=0.25,
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total de casos", "Incidência (/100 mil)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=casos,
            name="Casos",
            marker_color="#059669",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=incid,
            name="Incidência",
            marker_color="#2563eb",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=titulo or "Comparação epidemiológica",
        height=480,
        bargap=0.2,
    )
    fig.update_xaxes(tickangle=-25)
    return fig
