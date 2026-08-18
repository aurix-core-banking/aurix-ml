"""
Pipeline de Feature Engineering — extrai features do PostgreSQL e persiste no Feast.

Extrai dados de contas, transacoes e clientes do PostgreSQL (aurix_db),
aplica window functions para agregacoes temporais e salva no feature store
Feast (backed by S3/MinIO). Executado diariamente via Airflow.

Uso:
    python -m pipelines.feature_engineering --config ../ops/config/config.yaml
    python -m pipelines.feature_engineering --dry-run  # gera dados sinteticos sem PostgreSQL
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Mapeamento das tabelas do core bancario
TABELAS = {
    "contas": "contas",
    "transacoes": "transacoes",
    "clientes": "clientes",
}


def extrair_features_contas(conn_string: Optional[str] = None) -> pd.DataFrame:
    """Extrai features agregadas das contas bancarias.

    Join de contas + clientes com agregacoes de saldo e tempo de relacionamento.
    """
    if conn_string is None:
        logger.warning("conn_string nao fornecida — retornando DataFrame vazio")
        return pd.DataFrame()

    import sqlalchemy

    engine = sqlalchemy.create_engine(conn_string)

    query = """
    SELECT
        c.id_cliente,
        c.renda_mensal,
        c.idade,
        c.escolaridade,
        c.estado_civil,
        c.tipo_empregador,
        c.cidade,
        c.data_abertura,
        c.score_bureau,
        c.pessoas_residencia,
        c.possui_imovel,
        c.possui_veiculo,
        ct.saldo_atual,
        ct.saldo_medio_12m,
        ct.total_dividas,
        ct.total_financiado,
        ct.valor_parcela,
        ct.numero_operacoes_credito,
        EXTRACT(DAY FROM (NOW() - c.data_abertura)) / 30 AS tempo_conta_meses
    FROM clientes c
    INNER JOIN contas ct ON c.id_cliente = ct.id_cliente
    WHERE ct.ativa = true
    """

    df = pd.read_sql(query, engine)
    logger.info("Extraidas %d linhas de contas/clientes", len(df))
    return df


def extrair_features_transacoes(
    conn_string: Optional[str] = None,
    janela_dias: int = 90,
) -> pd.DataFrame:
    """Extrai agregacoes de transacoes usando window functions.

    Calcula por cliente:
    - Total de transacoes na janela
    - Valor medio e total
    - Frequencia por tipo
    - Dias desde a ultima transacao
    - Atrasos historicos e consultas recentes
    """
    if conn_string is None:
        logger.warning("conn_string nao fornecida — retornando DataFrame vazio")
        return pd.DataFrame()

    import sqlalchemy

    engine = sqlalchemy.create_engine(conn_string)
    data_corte = datetime.now() - timedelta(days=janela_dias)

    query = f"""
    SELECT
        t.id_cliente,
        COUNT(*) AS total_transacoes_{janela_dias}d,
        AVG(t.valor) AS valor_medio_transacao,
        SUM(t.valor) AS valor_total_transacoes,
        COUNT(CASE WHEN t.tipo = 'CREDITO' THEN 1 END) AS qtd_creditos,
        COUNT(CASE WHEN t.tipo = 'DEBITO' THEN 1 END) AS qtd_debitos,
        COUNT(CASE WHEN t.tipo = 'PIX' THEN 1 END) AS qtd_pix,
        MAX(t.data_transacao) AS data_ultima_transacao,
        EXTRACT(DAY FROM (NOW() - MAX(t.data_transacao))) AS dias_desde_ultima,
        SUM(CASE WHEN t.atrasado THEN 1 ELSE 0 END) AS atrasos_hist,
        COUNT(DISTINCT CASE WHEN t.tipo_consulta THEN t.data_transacao END) AS consultas_ultimo_6m
    FROM transacoes t
    WHERE t.data_transacao >= '{data_corte.strftime('%Y-%m-%d')}'
    GROUP BY t.id_cliente
    """

    df = pd.read_sql(query, engine)
    logger.info("Extraidas agregacoes de transacoes para %d clientes", len(df))
    return df


def join_features(
    df_contas: pd.DataFrame,
    df_transacoes: pd.DataFrame,
) -> pd.DataFrame:
    """Realiza o join das features de contas e transacoes.

    Merge por id_cliente, preenchendo valores faltantes com 0.
    """
    if df_transacoes.empty:
        logger.info("Sem dados de transacoes — usando apenas features de contas")
        return df_contas

    df = df_contas.merge(df_transacoes, on="id_cliente", how="left")
    colunas_transacao = [c for c in df_transacoes.columns if c != "id_cliente"]
    df[colunas_transacao] = df[colunas_transacao].fillna(0)
    logger.info("Join concluido: %d linhas, %d colunas", len(df), len(df.columns))
    return df


def calcular_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas de engenharia de dominio bancario.

    - saldo_ratio: saldo atual / saldo medio (estabilidade)
    - transacao_frequency: transacoes por mes de relacionamento
    - risk_score: score composto normalizado (0-1)
    - comprometimento_renda: dividas / renda
    - parcela_renda_ratio: parcela / renda
    """
    df = df.copy()

    # Saldo ratio
    df["saldo_ratio"] = df["saldo_atual"] / (df["saldo_medio_12m"] + 1)

    # Frequencia de transacoes
    meses = np.maximum(df.get("tempo_conta_meses", pd.Series(1, index=df.index)).fillna(1), 1)
    df["transacao_frequency"] = df.get("numero_operacoes_credito", 0) / meses

    # Comprometimento da renda
    df["comprometimento_renda"] = df["total_dividas"] / (df["renda_mensal"] + 1)

    # Parcela / renda
    df["parcela_renda_ratio"] = df["valor_parcela"] / (df["renda_mensal"] + 1)

    # Score de risco composto
    score_bureau_norm = df["score_bureau"].clip(0, 1000) / 1000.0
    atrasos_norm = df.get("atrasos_hist", pd.Series(0, index=df.index)).clip(0, 30) / 30.0
    comprometimento_norm = df["comprometimento_renda"].clip(0, 5) / 5.0
    df["risk_score"] = (
        (1 - score_bureau_norm) * 0.4
        + atrasos_norm * 0.3
        + comprometimento_norm * 0.3
    )

    logger.info("Features derivadas calculadas: saldo_ratio, transacao_frequency, risk_score")
    return df


def salvar_feast(
    df: pd.DataFrame,
    feature_view_name: str = "credit_features",
    repo_path: str = ".",
) -> None:
    """Salva as features no Feast feature store (backed by S3/MinIO).

    Registra o DataFrame como um Feature View no Feast.
    """
    try:
        from feast import Entity, FeatureView, Field, RepoConfig
        from feast.types import Float32, Int64, String
        from feast.infra.offline_stores.file_source import FileSource

        entity = Entity(name="id_cliente", join_keys=["id_cliente"])

        # Salva como Parquet para o Feast offline store
        output_dir = Path(repo_path) / "data" / "features"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{feature_view_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(output_path, index=False)

        source = FileSource(path=str(output_path), timestamp_field="event_timestamp")

        feature_view = FeatureView(
            name=feature_view_name,
            entities=[entity],
            schema=[
                Field(name="renda_mensal", dtype=Float32),
                Field(name="idade", dtype=Int64),
                Field(name="score_bureau", dtype=Int64),
                Field(name="saldo_medio_12m", dtype=Float32),
                Field(name="saldo_atual", dtype=Float32),
                Field(name="total_dividas", dtype=Float32),
                Field(name="total_financiado", dtype=Float32),
                Field(name="valor_parcela", dtype=Float32),
                Field(name="numero_operacoes_credito", dtype=Int64),
                Field(name="atrasos_hist", dtype=Int64),
                Field(name="consultas_ultimo_6m", dtype=Int64),
                Field(name="saldo_ratio", dtype=Float32),
                Field(name="transacao_frequency", dtype=Float32),
                Field(name="risk_score", dtype=Float32),
                Field(name="comprometimento_renda", dtype=Float32),
                Field(name="parcela_renda_ratio", dtype=Float32),
            ],
            source=source,
            online=True,
        )

        logger.info("Feature View '%s' registrada no Feast (%d linhas)", feature_view_name, len(df))
    except ImportError:
        logger.warning("Feast nao instalado — salvando features como Parquet apenas")
        output_dir = Path(repo_path) / "data" / "features"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{feature_view_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info("Features salvas em %s", output_path)
    except Exception as e:
        logger.warning("Falha ao registrar no Feast: %s — salvando como Parquet", e)
        output_dir = Path(repo_path) / "data" / "features"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{feature_view_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(output_path, index=False)


def gerar_dados_sinteticos(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Gera dados sinteticos para dry-run (sem PostgreSQL)."""
    try:
        from models.credit_risk_model_v2 import generate_credit_data_v2
        return generate_credit_data_v2(n_samples=n_samples, seed=seed)
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
        from credit_risk_model_v2 import generate_credit_data_v2
        return generate_credit_data_v2(n_samples=n_samples, seed=seed)


def executar_pipeline(
    conn_string: Optional[str] = None,
    janela_dias: int = 90,
    dry_run: bool = False,
    n_samples: int = 5000,
    repo_path: str = ".",
) -> pd.DataFrame:
    """Executa o pipeline completo de feature engineering.

    1. Extrai features do PostgreSQL (ou gera sinteticas em dry-run)
    2. Calcula features derivadas
    3. Salva no Feast feature store
    """
    logger.info("Iniciando pipeline de feature engineering (dry_run=%s)", dry_run)

    if dry_run:
        logger.info("Modo dry-run: gerando dados sinteticos")
        df = gerar_dados_sinteticos(n_samples=n_samples)
    else:
        df_contas = extrair_features_contas(conn_string)
        df_transacoes = extrair_features_transacoes(conn_string, janela_dias=janela_dias)
        df = join_features(df_contas, df_transacoes)

    if df.empty:
        logger.error("Nenhuma feature extraida — pipeline abortado")
        return df

    df = calcular_features_derivadas(df)

    # Adiciona timestamp para o Feast
    df["event_timestamp"] = datetime.now()

    salvar_feast(df, repo_path=repo_path)

    logger.info(
        "Pipeline concluido: %d clientes, %d features",
        len(df), len(df.columns),
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Pipeline de feature engineering")
    parser.add_argument("--config", default="../ops/config/config.yaml")
    parser.add_argument("--conn-string", help="String de conexao PostgreSQL")
    parser.add_argument("--janela-dias", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true", help="Gera dados sinteticos sem PostgreSQL")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--repo-path", default=".", help="Caminho do repo Feast")
    args = parser.parse_args()

    executar_pipeline(
        conn_string=args.conn_string,
        janela_dias=args.janela_dias,
        dry_run=args.dry_run,
        n_samples=args.n_samples,
        repo_path=args.repo_path,
    )


if __name__ == "__main__":
    main()
