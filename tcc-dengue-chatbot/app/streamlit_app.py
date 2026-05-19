import sys
from pathlib import Path

# Streamlit roda o script com cwd na raiz do projeto; garante import do pacote `app`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import chatbot, db, llm, queries


st.set_page_config(
    page_title="Chatbot Dengue — ETL + RAG",
    page_icon="🦟",
    layout="centered",
)

st.title("Chatbot Dengue — ETL + RAG")
# st.caption("ETL + PostgreSQL + RAG simples + OpenAI (MVP).")


def _safe_count() -> tuple[int, int, int]:
    try:
        with db.get_engine().connect() as conn:
            n_mun = conn.execute(queries.count_municipio_sql()).scalar() or 0
            n_fato = conn.execute(queries.count_fato_dengue_sql()).scalar() or 0
            n_log = conn.execute(queries.count_log_consulta_sql()).scalar() or 0
            return int(n_mun), int(n_fato), int(n_log)
    except Exception:
        return 0, 0, 0


with st.sidebar:
    st.subheader("Filtros")

    ok_pg, msg_pg = db.check_connection()
    st.metric("PostgreSQL", "OK" if ok_pg else "Erro", help=msg_pg)

    ok_openai, msg_openai = llm.openai_env_status()
    st.metric("OpenAI (env)", "OK" if ok_openai else "Atenção", help=msg_openai)

    n_mun, n_fato, n_log = _safe_count()
    st.caption(f"Municípios: {n_mun} · Fatos dengue: {n_fato} · Logs: {n_log}")

    # Dados para os selectboxes
    ufs_df = queries.listar_ufs() if ok_pg else pd.DataFrame(columns=["uf_sigla"])
    ufs = ufs_df["uf_sigla"].dropna().tolist() if not ufs_df.empty else []
    ufs = [str(x) for x in ufs]

    if ufs:
        uf = st.selectbox("UF", ["(qualquer)"] + ufs, index=0)
    else:
        uf = st.selectbox("UF", ["(sem dados)"], index=0, disabled=True)

    mun_options: list[str] = []
    if ok_pg and uf and uf not in ("(qualquer)", "(sem dados)"):
        mun_df = queries.listar_municipios_por_uf(uf)
        mun_options = mun_df["nome_municipio"].dropna().tolist() if not mun_df.empty else []
    if mun_options:
        municipio = st.selectbox("Município", ["(qualquer)"] + mun_options, index=0)
    else:
        municipio = st.selectbox("Município", ["(qualquer)"], index=0, disabled=True)

    anos_df = queries.listar_anos() if ok_pg else pd.DataFrame(columns=["ano"])
    anos = anos_df["ano"].dropna().tolist() if not anos_df.empty else []
    anos = [int(x) for x in anos]
    if anos:
        ano = st.selectbox("Ano", ["(qualquer)"] + anos, index=0)
    else:
        ano = st.selectbox("Ano", ["(sem dados)"], index=0, disabled=True)
    ano_val: int | None = None if ano == "(qualquer)" else (int(ano) if isinstance(ano, (int, float)) or (isinstance(ano, str) and ano.isdigit()) else None)

    semana_enabled = st.checkbox(
        "Semana epidemiológica opcional", value=False, disabled=(ano_val is None)
    )
    semana = None
    if semana_enabled:
        semana = st.number_input("Semana epidemiológica", min_value=1, max_value=53, value=1, step=1)


if not ok_pg:
    st.error("Não foi possível conectar ao PostgreSQL.")
    st.stop()

n_mun, n_fato, _ = _safe_count()
if n_fato <= 0:
    st.error(
        "Banco sem dados. Coloque os arquivos em data/raw e rode python -m etl.run_etl."
    )
    st.stop()


st.subheader("Pergunte")

suggestions = [
    "Qual foi a incidência de dengue em Palmas TO em 2025?",
    "Quantos casos de dengue houve em Palmas TO em 2025?",
    "Quais municípios do TO tiveram maior incidência em 2025?",
    "Compare Palmas e Araguaína em 2025.",
    "Compare a média do Paraná com a média de Palmas TO.",
    "A dengue está aumentando em Palmas TO nas últimas semanas?",
    "Palmas está acima da média do Tocantins em 2025?",
    "Qual foi a semana com mais casos em Palmas TO em 2025?",
]

if "submit" not in st.session_state:
    st.session_state.submit = False
if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False

pending_question = st.session_state.pop("pergunta_pendente", None)
if pending_question:
    st.session_state.pergunta_input = pending_question

filtros_ui: dict[str, object] = {}
if uf and uf not in ("(sem dados)", "(qualquer)"):
    filtros_ui["uf"] = uf
if municipio and municipio not in ("(sem dados)", "(qualquer)"):
    filtros_ui["municipio"] = municipio
if ano_val is not None:
    filtros_ui["ano"] = ano_val
if semana_enabled and semana is not None:
    filtros_ui["semana_epidemiologica"] = int(semana)

pergunta = st.text_input("Digite sua pergunta", key="pergunta_input")

col_send, col_spacer = st.columns([1, 4])
with col_send:
    if st.button("Enviar", use_container_width=True):
        st.session_state.submit = True

col1, col2 = st.columns(2)
for idx, sug in enumerate(suggestions):
    target_col = col1 if idx % 2 == 0 else col2
    with target_col:
        if st.button(sug, key=f"sug_{idx}", use_container_width=True):
            st.session_state.pergunta_pendente = sug
            st.session_state.auto_submit = True
            st.rerun()


def _render_result(result: dict) -> None:
    resposta = result.get("resposta", "")
    dados = result.get("dados")
    contexto = result.get("contexto") or []
    filtros_used = result.get("filtros") or {}
    tipo_consulta = result.get("tipo_consulta", "consulta")
    interpretacao = result.get("interpretacao") or {}

    st.markdown("### Resposta")
    st.write(resposta)

    st.markdown("### Dados")
    if isinstance(dados, pd.DataFrame):
        if not dados.empty:
            st.dataframe(dados, use_container_width=True)
        else:
            st.caption("Sem dados retornados para esta consulta.")
    elif isinstance(dados, dict):
        st.json(dados)
    else:
        st.caption("Dados não disponíveis neste formato.")

    if tipo_consulta in {"serie_semanal", "tendencia"} and isinstance(dados, pd.DataFrame) and not dados.empty:
        if "semana_epidemiologica" in dados.columns and "casos" in dados.columns:
            dfp = dados.sort_values(["ano", "semana_epidemiologica"]) if "ano" in dados.columns else dados.sort_values(["semana_epidemiologica"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dfp["semana_epidemiologica"],
                    y=dfp["casos"],
                    mode="lines+markers",
                    name="Casos",
                )
            )
            fig.update_layout(
                xaxis_title="Semana epidemiológica",
                yaxis_title="Casos",
                height=420,
            )
            st.markdown("### Gráfico (série semanal)")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Contexto RAG", expanded=False):
        if contexto:
            for i, item in enumerate(contexto, start=1):
                mun = item.get("municipio") or "—"
                uf = item.get("uf_sigla") or "—"
                ano = item.get("ano", "")
                sem = item.get("semana_epidemiologica", "")
                st.markdown(
                    f"**{i}. {mun}/{uf}** (ano {ano}, semana {sem})"
                )
                st.write(item.get("texto", ""))
                if i < len(contexto):
                    st.markdown("")
        else:
            st.caption("Nenhum contexto encontrado.")

    with st.expander("Filtros usados"):
        st.json(filtros_used)

    with st.expander("Interpretação da pergunta"):
        st.json(interpretacao)


if st.session_state.auto_submit:
    st.session_state.auto_submit = False
    st.session_state.submit = True

if st.session_state.submit:
    pergunta = (st.session_state.get("pergunta_input") or "").strip()
    if not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        try:
            with st.spinner("Consultando dados e gerando resposta..."):
                result = chatbot.answer_question(pergunta=pergunta, filtros_ui=filtros_ui)
            _render_result(result)
        except Exception as exc:
            st.error(f"Erro ao processar a pergunta: {exc}")
            st.exception(exc)
    st.session_state.submit = False
