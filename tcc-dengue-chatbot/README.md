# tcc-dengue-chatbot

MVP de chatbot sobre **dengue** com **Python**, **PostgreSQL**, **Streamlit**, **ETL**, **RAG** e **OpenAI API**, tudo em **Docker**.

## Arquitetura

- **Streamlit**: interface web do chatbot
- **PostgreSQL**: armazenamento analítico (dimensões/fatos) e tabelas de cartas/log
- **ETL (Python)**: lê dados em `data/raw/`, transforma e carrega no Postgres
- **RAG (MVP)**: recuperação simples no Postgres via busca textual (`ILIKE`) em `carta_de_fato`
- **OpenAI API**: geração de respostas com base em números calculados + contexto RAG

## Configuração (.env)

Crie o arquivo `.env` na raiz do projeto:

```bash
cp .env.example .env
```

Preencha:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=
```

## Como subir

```bash
docker compose up --build
```

## Onde colocar dados

- `data/raw/dengue`
- `data/raw/ibge`

## Como rodar o ETL

```bash
docker compose exec app python -m etl.run_etl
```

## Como acessar

http://localhost:8501

## Perguntas de teste

- Quantos casos de dengue houve em Palmas em 2024?
- Qual município teve mais casos em TO em 2024?
- Mostre a evolução semanal de Palmas em 2024.
- Qual foi a incidência de dengue em Palmas em 2024?
- Compare Palmas em 2023 e 2024.

## Limitações (MVP)

- Depende da qualidade/consistência da base **Sinan/Dengue**
- A população **IBGE** precisa estar disponível para incidência (quando existir)
- Respostas textuais usam **OpenAI API**
- Cálculos (casos/incidência) são feitos via **SQL/Python**; o LLM não calcula
- RAG inicial usa busca textual simples (`ILIKE`) em `carta_de_fato` (sem embeddings/Chroma no MVP)

## Checklist

- [ ] Docker sobe
- [ ] Postgres conecta
- [ ] Streamlit abre
- [ ] ETL carrega dados
- [ ] Chatbot responde
- [ ] Tabela aparece
- [ ] Gráfico aparece (quando for série semanal)
