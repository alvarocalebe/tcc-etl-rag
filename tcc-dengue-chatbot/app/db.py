from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Generator

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


class Municipio(Base):
    __tablename__ = "municipio"

    id_municipio_ibge: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    nome_municipio: Mapped[str] = mapped_column(String, nullable=False)
    uf_sigla: Mapped[str] = mapped_column(String(2), nullable=False)
    uf_codigo: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fatos_dengue: Mapped[list["FatoDengue"]] = relationship(back_populates="municipio")
    populacoes: Mapped[list["PopulacaoMunicipio"]] = relationship(back_populates="municipio")
    fatos_indicador: Mapped[list["FatoIndicador"]] = relationship(back_populates="municipio")


class Tempo(Base):
    __tablename__ = "tempo"
    __table_args__ = (
        UniqueConstraint("ano", "semana_epidemiologica", name="uq_tempo_ano_semana"),
    )

    id_tempo: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ano: Mapped[int] = mapped_column(Integer, nullable=False)
    semana_epidemiologica: Mapped[int] = mapped_column(Integer, nullable=False)
    mes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dt_inicio: Mapped[date | None] = mapped_column(Date, nullable=True)
    dt_fim: Mapped[date | None] = mapped_column(Date, nullable=True)

    fatos_dengue: Mapped[list["FatoDengue"]] = relationship(back_populates="tempo")
    fatos_indicador: Mapped[list["FatoIndicador"]] = relationship(back_populates="tempo")


class FatoDengue(Base):
    __tablename__ = "fato_dengue"
    __table_args__ = (
        UniqueConstraint("id_municipio_ibge", "id_tempo", name="uq_fato_dengue_mun_tempo"),
    )

    id_fato: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_municipio_ibge: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("municipio.id_municipio_ibge"), nullable=False
    )
    id_tempo: Mapped[int] = mapped_column(Integer, ForeignKey("tempo.id_tempo"), nullable=False)
    casos: Mapped[int] = mapped_column(Integer, nullable=False)
    fonte: Mapped[str | None] = mapped_column(String, nullable=True)
    data_extracao: Mapped[date | None] = mapped_column(Date, nullable=True)

    municipio: Mapped["Municipio"] = relationship(back_populates="fatos_dengue")
    tempo: Mapped["Tempo"] = relationship(back_populates="fatos_dengue")


class PopulacaoMunicipio(Base):
    __tablename__ = "populacao_municipio"
    __table_args__ = (
        UniqueConstraint("id_municipio_ibge", "ano", name="uq_pop_mun_mun_ano"),
    )

    id_pop: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_municipio_ibge: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("municipio.id_municipio_ibge"), nullable=False
    )
    ano: Mapped[int] = mapped_column(Integer, nullable=False)
    populacao: Mapped[int] = mapped_column(Integer, nullable=False)
    fonte: Mapped[str | None] = mapped_column(String, nullable=True)
    data_extracao: Mapped[date | None] = mapped_column(Date, nullable=True)

    municipio: Mapped["Municipio"] = relationship(back_populates="populacoes")


class FatoIndicador(Base):
    __tablename__ = "fato_indicador"
    __table_args__ = (
        UniqueConstraint("id_municipio_ibge", "id_tempo", name="uq_fato_ind_mun_tempo"),
    )

    id_indicador: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_municipio_ibge: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("municipio.id_municipio_ibge"), nullable=False
    )
    id_tempo: Mapped[int] = mapped_column(Integer, ForeignKey("tempo.id_tempo"), nullable=False)
    casos: Mapped[int] = mapped_column(Integer, nullable=False)
    populacao: Mapped[int | None] = mapped_column(Integer, nullable=True)
    incidencia_100k: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    fonte_casos: Mapped[str | None] = mapped_column(String, nullable=True)
    fonte_populacao: Mapped[str | None] = mapped_column(String, nullable=True)
    data_calculo: Mapped[date | None] = mapped_column(Date, nullable=True)

    municipio: Mapped["Municipio"] = relationship(back_populates="fatos_indicador")
    tempo: Mapped["Tempo"] = relationship(back_populates="fatos_indicador")
    cartas: Mapped[list["CartaDeFato"]] = relationship(back_populates="indicador")


class CartaDeFato(Base):
    __tablename__ = "carta_de_fato"

    id_carta: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_indicador: Mapped[int] = mapped_column(
        Integer, ForeignKey("fato_indicador.id_indicador"), nullable=False
    )
    texto: Mapped[str] = mapped_column(Text, nullable=False)
    municipio: Mapped[str | None] = mapped_column(String, nullable=True)
    uf_sigla: Mapped[str | None] = mapped_column(String(2), nullable=True)
    ano: Mapped[int | None] = mapped_column(Integer, nullable=True)
    semana_epidemiologica: Mapped[int | None] = mapped_column(Integer, nullable=True)
    incidencia_100k: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    casos: Mapped[int | None] = mapped_column(Integer, nullable=True)
    populacao: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fonte: Mapped[str | None] = mapped_column(String, nullable=True)
    data_extracao: Mapped[date | None] = mapped_column(Date, nullable=True)

    indicador: Mapped["FatoIndicador"] = relationship(back_populates="cartas")


class LogConsulta(Base):
    __tablename__ = "log_consulta"

    id_log: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pergunta: Mapped[str | None] = mapped_column(Text, nullable=True)
    resposta: Mapped[str | None] = mapped_column(Text, nullable=True)
    filtros: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, server_default=func.now()
    )


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


SessionLocal = sessionmaker(autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Sessão com commit/rollback automático."""
    session = SessionLocal(bind=get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> tuple[bool, str]:
    """Testa conexão com o Postgres."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Conectado"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
