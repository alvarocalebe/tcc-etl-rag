-- Schema analítico: municípios, tempo, fatos de dengue, população, indicadores, cartas e log.

CREATE TABLE IF NOT EXISTS municipio (
    id_municipio_ibge BIGINT PRIMARY KEY,
    nome_municipio    VARCHAR NOT NULL,
    uf_sigla          CHAR(2) NOT NULL,
    uf_codigo         INT NULL
);

CREATE TABLE IF NOT EXISTS tempo (
    id_tempo              SERIAL PRIMARY KEY,
    ano                   INT NOT NULL,
    semana_epidemiologica INT NOT NULL,
    mes                   INT NULL,
    dt_inicio             DATE NULL,
    dt_fim                DATE NULL,
    UNIQUE (ano, semana_epidemiologica)
);

CREATE TABLE IF NOT EXISTS fato_dengue (
    id_fato          SERIAL PRIMARY KEY,
    id_municipio_ibge BIGINT NOT NULL REFERENCES municipio (id_municipio_ibge),
    id_tempo         INT NOT NULL REFERENCES tempo (id_tempo),
    casos            INT NOT NULL,
    fonte            VARCHAR NULL,
    data_extracao    DATE NULL,
    UNIQUE (id_municipio_ibge, id_tempo)
);

CREATE TABLE IF NOT EXISTS populacao_municipio (
    id_pop            SERIAL PRIMARY KEY,
    id_municipio_ibge BIGINT NOT NULL REFERENCES municipio (id_municipio_ibge),
    ano               INT NOT NULL,
    populacao         INT NOT NULL,
    fonte             VARCHAR NULL,
    data_extracao     DATE NULL,
    UNIQUE (id_municipio_ibge, ano)
);

CREATE TABLE IF NOT EXISTS fato_indicador (
    id_indicador      SERIAL PRIMARY KEY,
    id_municipio_ibge BIGINT NOT NULL REFERENCES municipio (id_municipio_ibge),
    id_tempo          INT NOT NULL REFERENCES tempo (id_tempo),
    casos             INT NOT NULL,
    populacao         INT NULL,
    incidencia_100k   NUMERIC(12, 2) NULL,
    fonte_casos       VARCHAR NULL,
    fonte_populacao   VARCHAR NULL,
    data_calculo      DATE NULL,
    UNIQUE (id_municipio_ibge, id_tempo)
);

CREATE TABLE IF NOT EXISTS carta_de_fato (
    id_carta             SERIAL PRIMARY KEY,
    id_indicador         INT NOT NULL REFERENCES fato_indicador (id_indicador),
    texto                TEXT NOT NULL,
    municipio            VARCHAR NULL,
    uf_sigla             CHAR(2) NULL,
    ano                  INT NULL,
    semana_epidemiologica INT NULL,
    incidencia_100k      NUMERIC(12, 2) NULL,
    casos                INT NULL,
    populacao            INT NULL,
    fonte                VARCHAR NULL,
    data_extracao        DATE NULL
);

CREATE TABLE IF NOT EXISTS log_consulta (
    id_log      SERIAL PRIMARY KEY,
    pergunta    TEXT NULL,
    resposta    TEXT NULL,
    filtros     JSONB NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
