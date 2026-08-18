"""
Pipeline de Treino — carrega features do Feast, treina XGBoost/LightGBM com Optuna.

Pipeline completo de treino com:
- Carregamento de features do Feast feature store
- Train/test split (80/20)
- Treino com XGBoost e LightGBM
- Otimizacao de hiperparametros com Optuna
- Registro no MLflow com metrics e modelo
- Explicabilidade SHAP
- Validacao de regras de negocio (AUC > 0.8, KS > 0.3)

Uso:
    python -m pipelines.training_pipeline --config ../ops/config/config.yaml
    python -m pipelines.training_pipeline --dry-run --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regras de negocio para validacao do modelo
REGRAS_VALIDACAO = {
    "auc_minimo": 0.8,
    "ks_minimo": 0.3,
    "n_amostras_minimo": 100,
}

MODEL_NAME = "credit_risk_v2"
MODEL_VERSION = "2.0.0"
FRAMEWORK = "XGBoost"
EXPERIMENT_NAME = "aurix-credit-risk-v2"
REGISTRY_MODEL_NAME = "aurix-credit-risk-v2"


def carregar_features_feast(
    repo_path: str = ".",
    feature_view_name: str = "credit_features",
    dry_run: bool = False,
    n_samples: int = 5000,
) -> pd.DataFrame:
    """Carrega features do Feast feature store ou gera sinteticas em dry-run."""
    if dry_run:
        logger.info("Dry-run: gerando dados sinteticos")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
            from credit_risk_model_v2 import generate_credit_data_v2
            return generate_credit_data_v2(n_samples=n_samples)
        except ImportError:
            logger.error("credit_risk_model_v2 nao encontrado")
            return pd.DataFrame()

    try:
        from feast import FeatureStore
        store = FeatureStore(repo_path=repo_path)

        # Busca features historicas
        entity_df = pd.DataFrame({
            "id_cliente": np.arange(1, 10001),
            "event_timestamp": [datetime.now()] * 10000,
        })

        df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "credit_features:renda_mensal",
                "credit_features:idade",
                "credit_features:score_bureau",
                "credit_features:saldo_medio_12m",
                "credit_features:saldo_atual",
                "credit_features:total_dividas",
                "credit_features:total_financiado",
                "credit_features:valor_parcela",
                "credit_features:numero_operacoes_credito",
                "credit_features:atrasos_hist",
                "credit_features:consultas_ultimo_6m",
                "credit_features:saldo_ratio",
                "credit_features:transacao_frequency",
                "credit_features:risk_score",
            ],
        ).to_df()

        logger.info("Carregadas %d linhas do Feast", len(df))
        return df
    except Exception as e:
        logger.warning("Feast indisponivel: %s — gerando dados sinteticos", e)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
        from credit_risk_model_v2 import generate_credit_data_v2
        return generate_credit_data_v2(n_samples=n_samples)


def treinar_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Treina o modelo XGBoost com parametros default ou customizados."""
    import xgboost as xgb

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "tree_method": "hist",
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(
        **default_params,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def treinar_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Any:
    """Treina o modelo LightGBM."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM nao instalado — usando XGBoost")
        return treinar_xgboost(X_train, y_train, params, random_state)

    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def avaliar_modelo(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Avalia o modelo no holdout e retorna metricas completas."""
    from scipy import stats as scipy_stats
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # KS statistic
    ks_statistic = float(
        scipy_stats.ks_2samp(y_prob[y_test == 0], y_prob[y_test == 1]).statistic
    )

    return {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "acuracia": float(accuracy_score(y_test, y_pred)),
        "ks_statistic": ks_statistic,
        "n_amostras_teste": int(len(y_test)),
    }


def validar_regras_negocio(metricas: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Valida metricas contra regras de negocio.

    Returns:
        (aprovado, lista de violacoes)
    """
    violacoes = []

    if metricas["auc_roc"] < REGRAS_VALIDACAO["auc_minimo"]:
        violacoes.append(
            f"AUC-ROC {metricas['auc_roc']:.4f} < minimo {REGRAS_VALIDACAO['auc_minimo']}"
        )

    if metricas["ks_statistic"] < REGRAS_VALIDACAO["ks_minimo"]:
        violacoes.append(
            f"KS statistic {metricas['ks_statistic']:.4f} < minimo {REGRAS_VALIDACAO['ks_minimo']}"
        )

    n_amostras = metricas.get("n_amostras_teste", 0)
    if n_amostras < REGRAS_VALIDACAO["n_amostras_minimo"]:
        violacoes.append(
            f"Amostra teste {n_amostras} < minimo {REGRAS_VALIDACAO['n_amostras_minimo']}"
        )

    return len(violacoes) == 0, violacoes


def otimizar_hiperparametros(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 20,
    framework: str = "xgboost",
) -> Dict[str, Any]:
    """Otimiza hiperparametros com Optuna.

    Returns:
        Dict com melhores parametros e metrica
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna nao instalado — usando parametros default")
        return {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        if framework == "lightgbm":
            model = treinar_lightgbm(X_train, y_train, params)
        else:
            model = treinar_xgboost(X_train, y_train, params)

        from sklearn.metrics import roc_auc_score
        y_prob = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, y_prob))

    study = optuna.create_study(direction="maximize", study_name=f"{framework}_credit_risk")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "Optuna concluida: melhor AUC=%.4f (params=%s)",
        study.best_value, study.best_params,
    )
    return study.best_params


def gerar_shap_explanations(
    model: Any, X_test: pd.DataFrame, feature_columns: List[str],
) -> Optional[Dict[str, float]]:
    """Gera explicabilidade SHAP media por feature."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        importancia = np.abs(shap_values).mean(axis=0)
        return dict(sorted(
            zip(feature_columns, importancia),
            key=lambda kv: kv[1],
            reverse=True,
        ))
    except ImportError:
        logger.warning("SHAP nao instalado — pulando explicabilidade")
        return None


def registrar_mlflow(
    metricas: Dict[str, float],
    model: Any,
    feature_columns: List[str],
    params: Dict[str, Any],
    shap_explainer: Optional[Dict[str, float]],
    output_dir: Path,
    tracking_uri: str = "http://localhost:5000",
) -> Optional[str]:
    """Registra modelo e metricas no MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"train_credit_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id

            # Parametros
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("model_version", MODEL_VERSION)
            mlflow.log_param("framework", FRAMEWORK)
            mlflow.log_param("n_features", len(feature_columns))
            mlflow.log_param("feature_store", "Feast")
            for k, v in params.items():
                mlflow.log_param(f"hp_{k}", v)

            # Metricas
            for nome, valor in metricas.items():
                mlflow.log_metric(nome, valor)

            # Tags
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("model_version", MODEL_VERSION)
            mlflow.set_tag("framework", FRAMEWORK)
            mlflow.set_tag("pipeline", "training_pipeline_v2")

            # Artefato do modelo
            model_path = output_dir / f"{MODEL_NAME}.pkl"
            output_dir.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump({
                "model": model,
                "feature_columns": feature_columns,
                "metadata": {
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "training_date": datetime.now().isoformat(),
                },
            }, model_path)
            mlflow.log_artifact(str(model_path))

            # SHAP
            if shap_explainer:
                shap_path = output_dir / "shap_importance.json"
                with open(shap_path, "w") as f:
                    json.dump(shap_explainer, f, indent=2)
                mlflow.log_artifact(str(shap_path))

            # Registro do modelo
            try:
                mlflow.xgboost.log_model(
                    model, "model", registered_model_name=REGISTRY_MODEL_NAME,
                )
            except Exception:
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name=REGISTRY_MODEL_NAME,
                )

            logger.info("MLflow run registrado: %s", run_id)
            return run_id

    except ImportError:
        logger.warning("MLflow nao instalado — registro ignorado")
        return None
    except Exception as e:
        logger.warning("Falha no MLflow: %s", e)
        return None


def executar_pipeline(
    config: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    use_mlflow: bool = True,
    use_optuna: bool = True,
    output_dir: str = "models",
) -> Dict[str, Any]:
    """Executa o pipeline completo de treino.

    1. Carrega features do Feast (ou gera sinteticas em dry-run)
    2. Split train/test 80/20
    3. Treina XGBoost (e opcionalmente LightGBM)
    4. Otimiza hiperparametros com Optuna
    5. Valida regras de negocio
    6. Registra no MLflow
    7. Gera explicabilidade SHAP
    """
    config = config or {}
    train_cfg = config.get("credit_training", {})
    mlflow_cfg = config.get("mlflow", {})

    n_samples = train_cfg.get("sample_size", 5000)
    random_state = train_cfg.get("random_state", 42)
    test_size = config.get("training", {}).get("test_size", 0.2)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Carregar features
    logger.info("Etapa 1/7: Carregando features do Feast")
    df = carregar_features_feast(
        dry_run=dry_run, n_samples=n_samples,
    )
    if df.empty:
        return {"sucesso": False, "erro": "Nenhuma feature disponivel"}

    # 2. Preparar features e split
    logger.info("Etapa 2/7: Preparando features e split 80/20")
    from sklearn.model_selection import train_test_split

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
        from credit_risk_model_v2 import CreditRiskModelV2
    except ImportError:
        from models.credit_risk_model_v2 import CreditRiskModelV2

    model_wrapper = CreditRiskModelV2(random_state=random_state)
    target_col = train_cfg.get("target_column", "inadimplente")

    if target_col not in df.columns:
        # Gera target sintetico se nao existe (dry-run)
        df[target_col] = np.random.binomial(1, 0.15, len(df))

    X = model_wrapper.prepare_features(df)
    y = df[target_col].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Split adicional para validacao (Optuna)
    X_train_opt, X_val, y_train_opt, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train,
    )

    feature_columns = list(X.columns)

    # 3. Treinar XGBoost
    logger.info("Etapa 3/7: Treinando XGBoost")
    xgb_params = None
    if use_optuna:
        logger.info("  Otimizando hiperparametros com Optuna...")
        xgb_params = otimizar_hiperparametros(
            X_train_opt, y_train_opt, X_val, y_val, n_trials=20, framework="xgboost",
        )

    model_xgb = treinar_xgboost(X_train, y_train, xgb_params, random_state)

    # 4. Avaliar
    logger.info("Etapa 4/7: Avaliando modelo")
    metricas = avaliar_modelo(model_xgb, X_test, y_test)
    logger.info("Metricas: %s", json.dumps(metricas, indent=2))

    # 5. Validar regras de negocio
    logger.info("Etapa 5/7: Validando regras de negocio")
    aprovado, violacoes = validar_regras_negocio(metricas)
    if not aprovado:
        logger.warning("Violacoes de regras de negocio: %s", violacoes)
    else:
        logger.info("Todas as regras de negocio atendidas")

    # 6. SHAP
    logger.info("Etapa 6/7: Gerando explicabilidade SHAP")
    shap_explainer = gerar_shap_explanations(model_xgb, X_test, feature_columns)

    # 7. MLflow
    mlflow_run_id = None
    if use_mlflow:
        logger.info("Etapa 7/7: Registrando no MLflow")
        mlflow_run_id = registrar_mlflow(
            metricas, model_xgb, feature_columns, xgb_params or {},
            shap_explainer, output_path,
            tracking_uri=mlflow_cfg.get("tracking_uri", "http://localhost:5000"),
        )
    else:
        logger.info("Etapa 7/7: MLflow desabilitado — pulando registro")

    # Salva modelo localmente
    import joblib
    model_path = output_path / f"{MODEL_NAME}.pkl"
    joblib.dump({
        "model": model_xgb,
        "feature_columns": feature_columns,
        "metadata": {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "training_date": datetime.now().isoformat(),
            "metricas": metricas,
        },
    }, model_path)
    logger.info("Modelo salvo: %s", model_path)

    resultado = {
        "sucesso": True,
        "model_path": str(model_path),
        "metricas": metricas,
        "regras_aprovadas": aprovado,
        "violacoes": violacoes,
        "mlflow_run_id": mlflow_run_id,
        "shap_top_features": dict(list(shap_explainer.items())[:10]) if shap_explainer else None,
    }

    logger.info("Pipeline de treino concluido: %s", json.dumps({
        k: v for k, v in resultado.items() if k != "shap_top_features"
    }, indent=2))

    return resultado


def main():
    parser = argparse.ArgumentParser(description="Pipeline de treino credit_risk_v2")
    parser.add_argument("--config", default="../ops/config/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Dados sinteticos sem Feast")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()

    # Carrega config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    resultado = executar_pipeline(
        config=config,
        dry_run=args.dry_run,
        use_mlflow=not args.no_mlflow,
        use_optuna=not args.no_optuna,
        output_dir=args.output_dir,
    )

    if not resultado["sucesso"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
