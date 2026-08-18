"""
AUREUS ML - Pipeline de treino do modelo de Risco de Credito com governanca.

Treina o ``CreditRiskModel`` (XGBoost) sobre dados sinteticos do core bancario,
avalia com holdout, loga metricas e artefato no MLflow (registry com versao e
regime) e integra a governanca (regimes R1/R2/R3) para produzir a distribuicao
de decisoes governadas sobre a amostra.

Uso:
    python train_credit_pipeline.py --output-dir models --samples 5000
    python train_credit_pipeline.py --no-mlflow --regime r2 --governance-samples 100
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Garante imports do diretorio de modelos (mesmo padrao de ops/pipelines).
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Garante o pacote aurix_ml (governanca/llm) via shim no diretorio raiz.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from credit_risk_model import CreditRiskModel, generate_credit_data

# ---------------------------------------------------------------------------
# Constantes de versionamento/registro (padrao consistente entre os pipelines)
# ---------------------------------------------------------------------------
MODEL_NAME = "credit_risk"
MODEL_VERSION = "1.0.0"
ARTIFACT_FILE = "credit_risk_model.pkl"
REGISTRY_MODEL_NAME = "aurix-credit-risk"
EXPERIMENT_NAME = "aurix-credit-risk"
DEFAULT_SEED = 42
DEFAULT_RANDOM_STATE = 42
FRAMEWORK = "XGBoost"

# Regimes disponiveis na governanca (mapeamento nome -> classe do regime).
REGIMES = {
    "r1": "R1TextOnly",
    "r2": "R2Mechanical",
    "r3": "R3Adaptive",
    "auto": "R3Adaptive",
}

# Flags globais preenchidos pelo bloco de integracao de governanca.
GOVERNANCE_DISPONIVEL = False
try:
    from aurix_ml.governance import BankingCase, OperationType
    from aurix_ml.governance import R1TextOnly, R2Mechanical, R3Adaptive
    from aurix_ml.llm import create_llm
    GOVERNANCE_DISPONIVEL = True
except Exception as exc:  # pragma: no cover - ambiente sem aurix_ml
    print("Aviso: governanca indisponivel (%s) - pipeline segue sem regimes R1/R2/R3." % exc)


# ---------------------------------------------------------------------------
# MLflow (opcional - mesmo padrao de degracao de ops/pipelines/train_pipeline.py)
# ---------------------------------------------------------------------------
def _obter_mlflow():
    """Retorna modulo mlflow ou None se indisponivel (treino local sem registro)."""
    try:
        import mlflow
        return mlflow
    except Exception as exc:  # pragma: no cover - mlflow ausente
        print("Aviso: MLflow indisponivel (%s) - seguindo sem registro." % exc)
        return None


def _avaliar_modelo(model: CreditRiskModel, df: pd.DataFrame, random_state: int) -> dict:
    """Avalia o modelo em holdout 80/20 (AUC, acuracia, precisao, recall, F1)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X = model.prepare_features(df)
    y = df["inadimplente"].astype(int).values
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    proba = model.model.predict_proba(X_teste)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_teste, proba)),
        "acuracia": float(accuracy_score(y_teste, pred)),
        "precisao": float(precision_score(y_teste, pred, zero_division=0)),
        "recall": float(recall_score(y_teste, pred, zero_division=0)),
        "f1": float(f1_score(y_teste, pred, zero_division=0)),
        "n_amostras_teste": int(len(y_teste)),
    }


def _construir_caso(linha: pd.Series, caso_id: str) -> BankingCase:
    """Converte uma linha do DataFrame de credito em um caso governado."""
    renda = float(linha.get("renda_mensal", 0) or 0)
    total_financiado = float(linha.get("total_financiado", 0) or 0)
    comprometimento = float(linha.get("comprometimento_renda", 0) or 0)
    atrasos = int(linha.get("atrasos_hist", 0) or 0)
    score_bureau = int(np.clip(linha.get("score_bureau", 600) or 600, 0, 1000))

    # Score de risco interno (0..1) derivado do comprometimento e atrasos.
    risco = float(min(1.0, comprometimento / 2.0 + atrasos / 30.0))

    return BankingCase(
        case_id=caso_id,
        operation_type=OperationType.CREDIT_APPROVAL,
        amount=total_financiado,
        income=renda,
        client_score=score_bureau,
        risk_score=risco,
        context={
            "meses_inadimplencia_recente": atrasos,
            "comprometimento_renda": comprometimento,
            "numero_operacoes_credito": int(linha.get("numero_operacoes_credito", 0) or 0),
        },
    )


def _predicao_em_regimes(
    df: pd.DataFrame,
    regime_name: str,
    max_casos: int = 200,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Chama os regimes R1/R2/R3 da governanca sobre a amostra.

    Usa o provider ``mock`` por padrao (offline, determinístico e sem custo);
    o provider real pode ser definido pelas variaveis de ambiente
    ``AURIX_LLM_PROVIDER``/``AURIX_LLM_MODEL``.
    """
    if not GOVERNANCE_DISPONIVEL:
        raise RuntimeError("Governanca indisponivel - instale o pacote aurix_ml.")

    llm = create_llm({
        "provider": __import__("os").environ.get("AURIX_LLM_PROVIDER", "mock"),
        "model": __import__("os").environ.get("AURIX_LLM_MODEL", "governance"),
    })

    regime_map = {
        "R1TextOnly": R1TextOnly(llm),
        "R2Mechanical": R2Mechanical(llm),
        "R3Adaptive": R3Adaptive(llm),
    }
    regime = regime_map.get(regime_name, R3Adaptive(llm))

    linhas = []
    for idx, linha in df.head(max_casos).iterrows():
        caso = _construir_caso(linha, caso_id="TRAIN-CRED-%d" % idx)
        try:
            resultado = regime.decide(caso)
            decisao = resultado.decision.value
            regime_usado = resultado.regime
            audit_nonce = resultado.audit_nonce
        except Exception as exc:  # pragma: no cover - LLM fora do ar
            decisao = "ERRO"
            regime_usado = regime_name
            audit_nonce = ""
            print("Aviso: falha no regime %s para o caso %s (%s)" % (regime_name, caso.case_id, exc))
        linhas.append({
            "id_cliente": int(linha.get("id_cliente", idx)),
            "regime": regime_usado,
            "decisao": decisao,
            "audit_nonce": audit_nonce,
            "score_bureau": int(linha.get("score_bureau", 600) or 600),
            "inadimplente": int(linha.get("inadimplente", 0) or 0),
        })
    return pd.DataFrame(linhas)


def _registrar_mlflow(
    mlflow,
    model: CreditRiskModel,
    metricas: dict,
    artefato_path: str,
    governanca_path: str,
    args: argparse.Namespace,
) -> str:
    """Loga parametros, metricas, tags de versao/regime e registra o modelo."""
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="train_credit_%s" % datetime.now().strftime("%Y%m%d_%H%M%S")) as run:
        run_id = run.info.run_id

        mlflow.log_param("sample_size", args.samples)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("target_column", "inadimplente")
        mlflow.log_param("regime_governanca", args.regime)
        mlflow.log_param("governance_samples", args.governance_samples)

        for nome, valor in metricas.items():
            mlflow.log_metric(nome, valor)

        # Tags de versionamento e regime (padrao consistente entre pipelines).
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("model_version", MODEL_VERSION)
        mlflow.set_tag("framework", FRAMEWORK)
        mlflow.set_tag("regime", args.regime)
        mlflow.set_tag("governance_regimes", "r1_text_only,r2_mechanical,r3_adaptive")

        # Artefato no mesmo formato dos demais pipelines (.pkl + metadados).
        Path(artefato_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(artefato_path))
        mlflow.log_artifact(str(artefato_path))

        if governanca_path is not None and Path(governanca_path).exists():
            mlflow.log_artifact(str(governanca_path))

        try:
            mlflow.xgboost.log_model(
                model.model,
                "model",
                registered_model_name=args.registry_model_name,
            )
        except Exception:
            # Fallback: flavor sklearn para estimadores com API sklearn.
            mlflow.sklearn.log_model(
                model.model,
                "model",
                registered_model_name=args.registry_model_name,
            )

    return run_id


def main():
    parser = argparse.ArgumentParser(description="Pipeline de treino do modelo de risco de credito com governanca")
    parser.add_argument("--output-dir", type=str, default="models", help="Diretorio de saida do artefato .pkl")
    parser.add_argument("--samples", type=int, default=5000, help="Numero de amostras sinteticas")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Semente global de reprodutibilidade")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Semente do modelo XGBoost")
    parser.add_argument("--regime", choices=["r1", "r2", "r3", "auto"], default="r3",
                        help="Regime de governanca usado na predicao em regimes")
    parser.add_argument("--governance-samples", type=int, default=200,
                        help="Tamanho da amostra submetida aos regimes R1/R2/R3")
    parser.add_argument("--skip-governance", action="store_true", help="Pula a integracao com a governanca")
    parser.add_argument("--no-mlflow", action="store_true", help="Desabilita registro no MLflow")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--experiment-name", type=str, default=EXPERIMENT_NAME)
    parser.add_argument("--registry-model-name", type=str, default=REGISTRY_MODEL_NAME)
    args = parser.parse_args()

    # Reproductibilidade global do script (mesmo padrao dos demais pipelines).
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artefato_path = out / ARTIFACT_FILE

    print("Gerando dados de credito (n=%d, seed=%d)..." % (args.samples, args.seed))
    df = generate_credit_data(n_samples=args.samples, seed=args.seed)

    print("Treinando modelo de risco de credito (XGBoost)...")
    model = CreditRiskModel(random_state=args.random_state)
    model.train(df, target_column="inadimplente")

    metricas = _avaliar_modelo(model, df, random_state=args.seed)
    print("Metricas holdout: %s" % json.dumps(metricas, indent=2))

    governanca_path = None
    if not args.skip_governance and GOVERNANCE_DISPONIVEL:
        print("Integrando governanca (regime=%s, amostras=%d)..." % (
            args.regime, args.governance_samples))
        regime_classe = REGIMES.get(args.regime, "R3Adaptive")
        predicoes = _predicao_em_regimes(
            df, regime_classe, max_casos=args.governance_samples, seed=args.seed
        )
        governanca_path = out / "governance_predicoes.csv"
        predicoes.to_csv(governanca_path, index=False)
        distribuicao = Counter(predicoes["decisao"])
        print("Distribuicao de decisoes governadas: %s" % dict(distribuicao))
        for decisao, qtd in sorted(distribuicao.items()):
            metricas["governanca_%s" % decisao] = int(qtd)
        metricas["governanca_total"] = int(len(predicoes))
    elif not args.skip_governance:
        print("Aviso: governanca indisponivel - pulando integracao com regimes R1/R2/R3.")

    # Metadados de versao/regime embutidos no artefato (usados pelo serving).
    model.metadata["model_name"] = MODEL_NAME
    model.metadata["model_version"] = MODEL_VERSION
    model.metadata["framework"] = FRAMEWORK
    model.metadata["regime"] = args.regime

    model.save_model(str(artefato_path))
    print("Artefato salvo: %s" % artefato_path)

    run_id = None
    if not args.no_mlflow:
        mlflow = _obter_mlflow()
        if mlflow is not None:
            run_id = _registrar_mlflow(
                mlflow, model, metricas, str(artefato_path), governanca_path, args
            )
            print("Registrado no MLflow (run=%s, registry=%s)" % (run_id, args.registry_model_name))
        else:
            print("Aviso: MLflow indisponivel - artefato salvo sem registro.")
    else:
        print("Aviso: MLflow desabilitado via --no-mlflow.")

    print("Pipeline de treino de credito concluido.")

    return {
        "mlflow_run_id": run_id,
        "model_path": str(artefato_path),
        "metrics": metricas,
    }


if __name__ == "__main__":
    main()
