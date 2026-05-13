-- Índices para filtros por município, UF, ano, semana e cartas

CREATE INDEX IF NOT EXISTS idx_municipio_uf ON municipio (uf_sigla);
CREATE INDEX IF NOT EXISTS idx_municipio_nome ON municipio (nome_municipio);

CREATE INDEX IF NOT EXISTS idx_tempo_ano ON tempo (ano);
CREATE INDEX IF NOT EXISTS idx_tempo_ano_semana ON tempo (ano, semana_epidemiologica);

CREATE INDEX IF NOT EXISTS idx_fato_dengue_municipio ON fato_dengue (id_municipio_ibge);
CREATE INDEX IF NOT EXISTS idx_fato_dengue_tempo ON fato_dengue (id_tempo);
CREATE INDEX IF NOT EXISTS idx_fato_dengue_ano_semana_mun ON fato_dengue (id_tempo, id_municipio_ibge);

CREATE INDEX IF NOT EXISTS idx_pop_mun_municipio ON populacao_municipio (id_municipio_ibge);
CREATE INDEX IF NOT EXISTS idx_pop_mun_ano ON populacao_municipio (ano);

CREATE INDEX IF NOT EXISTS idx_fato_ind_municipio ON fato_indicador (id_municipio_ibge);
CREATE INDEX IF NOT EXISTS idx_fato_ind_tempo ON fato_indicador (id_tempo);
CREATE INDEX IF NOT EXISTS idx_fato_ind_incidencia ON fato_indicador (incidencia_100k);

CREATE INDEX IF NOT EXISTS idx_carta_indicador ON carta_de_fato (id_indicador);
CREATE INDEX IF NOT EXISTS idx_carta_uf_ano_semana
    ON carta_de_fato (uf_sigla, ano, semana_epidemiologica);
CREATE INDEX IF NOT EXISTS idx_carta_municipio_nome ON carta_de_fato (municipio);

CREATE INDEX IF NOT EXISTS idx_log_consulta_created_at ON log_consulta (created_at DESC);
