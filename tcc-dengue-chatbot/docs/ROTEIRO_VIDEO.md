# Roteiro de vídeo — Chatbot Dengue (TCC)

Demonstração em **4 atos** (~5–8 min): subir a aplicação, importar dados (ETL), provar carga no PostgreSQL e usar o chatbot no Streamlit.

---

## Checklist antes de gravar

| Item | Verificação |
|------|-------------|
| Docker | Docker Desktop em execução |
| `.env` | `OPENAI_API_KEY` preenchida (texto das respostas) |
| Dados dengue | `data/raw/dengue/DENGBR25.csv` presente |
| Dados IBGE (opcional) | Arquivo em `data/raw/ibge/` — necessário para **incidência** |
| Terminal | Fonte grande, tema escuro |
| Navegador | http://localhost:8501 em tela cheia |
| ETL | CSV com ~1,5M linhas — pode levar **vários minutos** (use time-lapse ou corte) |

---

## Ordem das janelas na gravação

```
Terminal 1 → docker compose up (deixar rodando)
Terminal 2 → ETL e, depois, consultas no PostgreSQL
Navegador   → Streamlit: antes (vazio) → depois (com dados) → perguntas
```

**Narrativa:** dados brutos → ETL → prova no banco → chatbot em linguagem natural.

---

## Ato 1 — Subir a aplicação (~1 min)

### Comandos

```bash
cd /Users/alvarocalebe/FACULDADE/TCC/tcc-dengue-chatbot
docker compose up --build
```

### O que mostrar

- Container `tcc-dengue-postgres` healthy
- Container `tcc-dengue-app` com Streamlit
- Abrir http://localhost:8501
- Sidebar: **PostgreSQL OK**; contadores **Municípios / Fatos dengue** em 0 (se banco novo)

### Fala (sugestão)

> A stack sobe com Docker: PostgreSQL para a base analítica e Streamlit para a interface do chatbot. O schema é criado automaticamente na primeira subida.

---

## Ato 2 — Importação de dados (ETL) (~2–4 min)

Abrir **segundo terminal** (manter o `docker compose up` no primeiro).

### Comandos

```bash
cd /Users/alvarocalebe/FACULDADE/TCC/tcc-dengue-chatbot
docker compose exec app python -m etl.run_etl
```

### Saída esperada (destacar na tela)

```
Linhas lidas (dengue): ...
Linhas lidas (IBGE): ...
Municípios: ...
Fatos dengue: ...
Indicadores: ...
Cartas: ...
Sucesso: ETL finalizado e dados carregados no PostgreSQL.
```

### Fala (sugestão)

> O ETL lê as notificações de dengue do Sinan, normaliza municípios e semanas epidemiológicas, calcula indicadores e carrega o modelo dimensional no PostgreSQL. As cartas de fato alimentam o RAG.

### Dica de edição

Gravar só o início e o fim do comando; no meio, legenda: *“Após alguns minutos…”*.

---

## Ato 3 — Dados no PostgreSQL (~1,5 min)

### Opção A — Uma linha (rápido, sem entrar no psql)

```bash
docker compose exec postgres psql -U dengue_user -d dengue_db -c "
SELECT
    (SELECT COUNT(*) FROM municipio) AS municipios,
    (SELECT COUNT(*) FROM fato_dengue) AS fatos_casos,
    (SELECT COALESCE(SUM(casos), 0) FROM fato_dengue) AS total_casos;
"
```

### Opção B — Sessão interativa

```bash
docker compose exec postgres psql -U dengue_user -d dengue_db
```

#### Query 1 — Resumo geral (abrir o psql com isso)

```sql
SELECT
    (SELECT COUNT(*) FROM municipio)          AS municipios,
    (SELECT COUNT(*) FROM tempo)              AS periodos,
    (SELECT COUNT(*) FROM fato_dengue)        AS fatos_casos,
    (SELECT COUNT(*) FROM fato_indicador)     AS indicadores,
    (SELECT COUNT(*) FROM carta_de_fato)      AS cartas_rag,
    (SELECT COALESCE(SUM(casos), 0) FROM fato_dengue) AS total_casos;
```

Critério de sucesso: `municipios`, `fatos_casos` e `total_casos` **maiores que zero**.

#### Query 2 — Amostra: Palmas/TO

```sql
SELECT
    m.nome_municipio,
    m.uf_sigla,
    t.ano,
    t.semana_epidemiologica,
    fd.casos
FROM fato_dengue fd
JOIN municipio m ON m.id_municipio_ibge = fd.id_municipio_ibge
JOIN tempo t ON t.id_tempo = fd.id_tempo
WHERE m.nome_municipio ILIKE 'Palmas%'
  AND m.uf_sigla = 'TO'
ORDER BY t.ano DESC, t.semana_epidemiologica DESC
LIMIT 10;
```

#### Query 3 — Top municípios por casos (ano)

```sql
SELECT
    m.nome_municipio,
    m.uf_sigla,
    t.ano,
    SUM(fd.casos) AS casos
FROM fato_dengue fd
JOIN municipio m ON m.id_municipio_ibge = fd.id_municipio_ibge
JOIN tempo t ON t.id_tempo = fd.id_tempo
WHERE t.ano = 2025
GROUP BY m.nome_municipio, m.uf_sigla, t.ano
ORDER BY casos DESC
LIMIT 5;
```

Ajuste o ano (`2025`) se a base tiver outro recorte principal.

### Fala (sugestão)

> Os dados não ficam só no CSV: estão no modelo com dimensões de município e tempo, fatos de casos, indicadores e cartas para busca textual no RAG.

### Conexão (referência)

| Campo | Valor |
|-------|--------|
| Host | `localhost` |
| Porta | `5432` |
| Banco | `dengue_db` |
| Usuário | `dengue_user` |
| Senha | `dengue_pass` |

---

## Ato 4 — Chatbot no Streamlit (~2–3 min)

Recarregar http://localhost:8501. A sidebar deve mostrar **Municípios** e **Fatos dengue** > 0.

### Perguntas sugeridas (3 para o vídeo)

| # | Pergunta | Destaque na tela |
|---|----------|------------------|
| 1 | Quantos casos de dengue houve em Palmas TO em 2025? | Resposta + seção **Dados** (tabela) |
| 2 | Mostre a evolução semanal de Palmas TO em 2025. | Tabela + **Gráfico (série semanal)** |
| 3 | Compare Palmas e Araguaína em 2025. | Comparação; expander **Interpretação da pergunta** |

### Outras perguntas (botões da interface)

- Qual foi a incidência de dengue em Palmas TO em 2025?
- Quais municípios do TO tiveram maior incidência em 2025?
- Compare a média do Paraná com a média de Palmas TO.
- A dengue está aumentando em Palmas TO nas últimas semanas?
- Palmas está acima da média do Tocantins em 2025?
- Qual foi a semana com mais casos em Palmas TO em 2025?

### O que abrir em uma pergunta (detalhe técnico)

- **Contexto RAG** — cartas recuperadas do Postgres
- **Filtros usados** — município, UF, ano vindos da UI ou da pergunta
- **Interpretação da pergunta** — intenção e entidades (NLU)

### Fala (sugestão)

> A pergunta é interpretada em linguagem natural, os números vêm de consultas SQL no PostgreSQL, o RAG traz contexto das cartas de fato, e o modelo de linguagem só redige a resposta — não inventa os totais.

---

## Demo do zero (opcional)

Para gravar banco vazio → ETL → cheio:

```bash
docker compose down -v
docker compose up --build
# Em outro terminal:
docker compose exec app python -m etl.run_etl
```

**Atenção:** `-v` apaga o volume do Postgres.

---

## Slide de arquitetura (5–10 s)

Mostrar `docs/der_base_analitica.png` ou `docs/der_base_analitica.svg`:

**ETL** → **PostgreSQL** (dimensões/fatos/cartas) → **Streamlit** (NLU + SQL + RAG + OpenAI).

---

## Encerramento — script de 30 segundos

> Implementei um pipeline ETL que importa notificações de dengue para um modelo dimensional no PostgreSQL. O chatbot interpreta perguntas em linguagem natural, executa consultas estruturadas e complementa com contexto RAG, exibindo tabelas e gráficos no Streamlit. O objetivo é tornar dados epidemiológicos acessíveis para consulta interativa, separando cálculo no banco da geração de texto pela API de linguagem.

---

## Roteiro falado completo (palavra por palavra)

Use como teleprompter; adapte números/anos ao que aparecer na sua base.

### Abertura (15 s)

> Olá. Neste vídeo apresento o chatbot de dengue do meu TCC: um MVP que integra ETL, PostgreSQL, recuperação de contexto e interface em Streamlit.

### Ato 1 (30 s)

> Primeiro subo o ambiente com Docker Compose. O Postgres recebe o schema analítico na inicialização, e o app sobe o Streamlit na porta 8501. Na barra lateral dá para ver se a conexão com o banco está ok; antes da importação, ainda não há fatos carregados.

### Ato 2 (45 s)

> Em seguida executo o ETL dentro do container da aplicação. Ele lê o arquivo nacional de dengue em data/raw, trata municípios e semanas epidemiológicas, gera indicadores e cartas para o RAG, e grava tudo no Postgres. O processo pode demorar por causa do volume do CSV — aqui aparecem as contagens de municípios, fatos e cartas ao final.

### Ato 3 (40 s)

> Para comprovar a carga, consulto o banco diretamente. Esta query resume quantos municípios, fatos e casos existem. Depois mostro um recorte de Palmas no Tocantins e um ranking de municípios por casos no ano — provando que os dados estão estruturados e consultáveis via SQL.

### Ato 4 (60 s)

> Volto ao Streamlit, atualizo a página e os contadores na lateral já refletem os dados importados. Faço três perguntas: total de casos em Palmas, evolução semanal com gráfico, e comparação entre dois municípios. Os valores vêm do banco; a resposta em texto usa a API configurada no ambiente. Nos expanders dá para ver a interpretação da pergunta, os filtros e o contexto RAG.

### Fechamento (20 s)

> Em resumo: dados brutos do Sinan viram base analítica no PostgreSQL, e o usuário consulta essa base em linguagem natural, com tabelas e gráficos na interface. Obrigado.

---

## Troubleshooting na gravação

| Problema | Ação |
|----------|------|
| Sidebar com 0 fatos após ETL | Recarregar a página; conferir Query 1 no Postgres |
| Erro OpenAI | Verificar `.env`; ainda é possível mostrar **Dados** e gráfico |
| Incidência vazia | Colocar população IBGE em `data/raw/ibge/` e rodar ETL de novo |
| ETL muito lento | Gravar time-lapse; mencionar volume do DENGBR25 |
| Porta 5432 ocupada | Parar outro Postgres local ou alterar porta no `docker-compose.yml` |

---

## Referências no repositório

- README: comandos de subida e ETL
- `sql/01_schema.sql`: modelo de dados
- `app/streamlit_app.py`: sugestões de perguntas na UI
- `docs/der_base_analitica.puml`: diagrama ER
