"""
Detector de Drift — PSI, KS test, Chi-squared para features numericas e categoricas.

Monitora tres tipos de drift:
- Feature drift: mudanca na distribuicao das features de entrada
- Prediction drift: mudanca na distribuicao das predicoes
- Concept drift: mudanca na relacao features -> target

Thresholds:
- PSI > 0.2 = alerta, PSI > 0.5 = critico
- KS p-value < 0.05 = drift significativo
- Chi-squared p-value < 0.05 = drift categorico significativo
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EPSILON = 1e-6
PSI_ALERTA = 0.2
PSI_CRITICO = 0.5
KS_PVALUE_LIMITE = 0.05
CHI2_PVALUE_LIMITE = 0.05


class DriftDetector:
    """Detector de drift com PSI, KS test e Chi-squared."""

    def __init__(
        self,
        psi_alerta: float = PSI_ALERTA,
        psi_critico: float = PSI_CRITICO,
        ks_pvalue_limite: float = KS_PVALUE_LIMITE,
        chi2_pvalue_limite: float = CHI2_PVALUE_LIMITE,
    ):
        self.psi_alerta = psi_alerta
        self.psi_critico = psi_critico
        self.ks_pvalue_limite = ks_pvalue_limite
        self.chi2_pvalue_limite = chi2_pvalue_limite

    def calcular_psi(
        self, esperado: np.ndarray, atual: np.ndarray, n_bins: int = 10,
    ) -> float:
        """Calcula o PSI entre duas distribuicoes."""
        esperado = np.asarray(esperado, dtype=float)
        atual = np.asarray(atual, dtype=float)
        esperado = esperado[~np.isnan(esperado)]
        atual = atual[~np.isnan(atual)]

        if len(esperado) < n_bins or len(atual) < n_bins:
            return 0.0

        bins = np.quantile(esperado, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        counts_esp, _ = np.histogram(esperado, bins=bins)
        counts_atu, _ = np.histogram(atual, bins=bins)
        prop_esp = np.clip(counts_esp / counts_esp.sum(), EPSILON, None)
        prop_atu = np.clip(counts_atu / counts_atu.sum(), EPSILON, None)
        return float(np.sum((prop_atu - prop_esp) * np.log(prop_atu / prop_esp)))

    def teste_ks(self, esperado: np.ndarray, atual: np.ndarray) -> Dict[str, float]:
        """Teste de Kolmogorov-Smirnov entre duas distribuicoes."""
        esperado = np.asarray(esperado, dtype=float)[~np.isnan(np.asarray(esperado, dtype=float))]
        atual = np.asarray(atual, dtype=float)[~np.isnan(np.asarray(atual, dtype=float))]
        if len(esperado) < 5 or len(atual) < 5:
            return {"statistic": 0.0, "p_value": 1.0}
        resultado = scipy_stats.ks_2samp(esperado, atual)
        return {"statistic": float(resultado.statistic), "p_value": float(resultado.pvalue)}

    def teste_chi2(self, esperado: pd.Series, atual: pd.Series) -> Dict[str, float]:
        """Teste Chi-quadrado para distribuicao de categorias."""
        contagem_esp = esperado.value_counts()
        contagem_atu = atual.value_counts()
        todas_cats = sorted(set(contagem_esp.index) | set(contagem_atu.index))
        esp_arr = np.array([contagem_esp.get(c, 0) for c in todas_cats], dtype=float)
        atu_arr = np.array([contagem_atu.get(c, 0) for c in todas_cats], dtype=float)
        mascara = (esp_arr + atu_arr) > 0
        esp_arr, atu_arr = esp_arr[mascara], atu_arr[mascara]
        if len(esp_arr) < 2 or esp_arr.sum() < 5:
            return {"statistic": 0.0, "p_value": 1.0, "drift_detectado": False}
        resultado = scipy_stats.chi2_contingency(np.array([esp_arr, atu_arr]))
        return {
            "statistic": float(resultado[0]),
            "p_value": float(resultado[1]),
            "drift_detectado": float(resultado[1]) < self.chi2_pvalue_limite,
        }

    def _classificar_nivel(self, psi: float, ks_pvalue: float) -> str:
        """Classifica o nivel de drift."""
        if psi > self.psi_critico:
            return "critico"
        if psi > self.psi_alerta:
            return "alerta"
        if ks_pvalue < self.ks_pvalue_limite:
            return "atencao"
        return "ok"

    def detectar_feature_drift(
        self,
        df_referencia: pd.DataFrame,
        df_atual: pd.DataFrame,
        features_numericas: Optional[List[str]] = None,
        features_categoricas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detecta feature drift em todas as features."""
        if features_numericas is None:
            features_numericas = [
                c for c in df_referencia.columns
                if df_referencia[c].dtype in [np.float64, np.int64, float, int]
                and c not in ("id_cliente", "inadimplente", "event_timestamp")
            ]
        if features_categoricas is None:
            features_categoricas = [
                c for c in df_referencia.columns
                if df_referencia[c].dtype == object
                and c not in ("id_cliente", "event_timestamp")
            ]

        resultado: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "features_numericas": {},
            "features_categoricas": {},
            "psi_geral": 0.0,
            "drift_detectado": False,
            "nivel": "ok",
        }
        psis = []

        for col in features_numericas:
            if col not in df_atual.columns:
                continue
            ref = df_referencia[col].dropna().values
            cur = df_atual[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue
            psi = self.calcular_psi(ref, cur)
            ks = self.teste_ks(ref, cur)
            nivel = self._classificar_nivel(psi, ks["p_value"])
            resultado["features_numericas"][col] = {
                "psi": round(psi, 6),
                "ks_statistic": round(ks["statistic"], 6),
                "ks_p_value": round(ks["p_value"], 6),
                "nivel": nivel,
            }
            psis.append(psi)

        for col in features_categoricas:
            if col not in df_atual.columns:
                continue
            chi2 = self.teste_chi2(df_referencia[col], df_atual[col])
            nivel = "critico" if chi2["drift_detectado"] else "ok"
            resultado["features_categoricas"][col] = {
                "chi2_statistic": round(chi2["statistic"], 6),
                "chi2_p_value": round(chi2["p_value"], 6),
                "drift_detectado": chi2["drift_detectado"],
                "nivel": nivel,
            }

        if psis:
            resultado["psi_geral"] = round(float(np.mean(psis)), 6)
            resultado["drift_detectado"] = resultado["psi_geral"] > self.psi_alerta
            nivel_max = "ok"
            for nivel in [f["nivel"] for f in resultado["features_numericas"].values()]:
                if nivel == "critico":
                    nivel_max = "critico"
                    break
                if nivel == "alerta" and nivel_max != "critico":
                    nivel_max = "alerta"
            resultado["nivel"] = nivel_max

        return resultado

    def detectar_prediction_drift(
        self,
        predicoes_referencia: np.ndarray,
        predicoes_atuais: np.ndarray,
    ) -> Dict[str, Any]:
        """Detecta drift na distribuicao das predicoes."""
        psi = self.calcular_psi(predicoes_referencia, predicoes_atuais)
        ks = self.teste_ks(predicoes_referencia, predicoes_atuais)
        nivel = self._classificar_nivel(psi, ks["p_value"])

        return {
            "timestamp": datetime.now().isoformat(),
            "psi": round(psi, 6),
            "ks_statistic": round(ks["statistic"], 6),
            "ks_p_value": round(ks["p_value"], 6),
            "drift_detectado": psi > self.psi_alerta,
            "nivel": nivel,
        }

    def detectar_concept_drift(
        self,
        y_referencia: np.ndarray,
        y_atuais: np.ndarray,
        predicoes_referencia: np.ndarray,
        predicoes_atuais: np.ndarray,
    ) -> Dict[str, Any]:
        """Detecta concept drift (mudanca na relacao features -> target).

        Compara a distribuicao de erros entre referencia e atual.
        """
        erros_ref = np.abs(y_referencia - predicoes_referencia)
        erros_atu = np.abs(y_atuais - predicoes_atuais)
        psi = self.calcular_psi(erros_ref, erros_atu)
        ks = self.teste_ks(erros_ref, erros_atu)
        nivel = self._classificar_nivel(psi, ks["p_value"])

        return {
            "timestamp": datetime.now().isoformat(),
            "psi_erros": round(psi, 6),
            "ks_erros_statistic": round(ks["statistic"], 6),
            "ks_erros_p_value": round(ks["p_value"], 6),
            "drift_detectado": psi > self.psi_alerta,
            "nivel": nivel,
        }

    def gerar_relatorio_completo(
        self,
        df_referencia: pd.DataFrame,
        df_atual: pd.DataFrame,
        predicoes_referencia: Optional[np.ndarray] = None,
        predicoes_atuais: Optional[np.ndarray] = None,
        y_referencia: Optional[np.ndarray] = None,
        y_atuais: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Gera relatorio completo de drift (feature + prediction + concept)."""
        relatorio: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "feature_drift": self.detectar_feature_drift(df_referencia, df_atual),
        }

        if predicoes_referencia is not None and predicoes_atuais is not None:
            relatorio["prediction_drift"] = self.detectar_prediction_drift(
                predicoes_referencia, predicoes_atuais,
            )

        if all(v is not None for v in [y_referencia, y_atuais, predicoes_referencia, predicoes_atuais]):
            relatorio["concept_drift"] = self.detectar_concept_drift(
                y_referencia, y_atuais, predicoes_referencia, predicoes_atuais,
            )

        # Nivel geral
        niveis = [relatorio["feature_drift"]["nivel"]]
        if "prediction_drift" in relatorio:
            niveis.append(relatorio["prediction_drift"]["nivel"])
        if "concept_drift" in relatorio:
            niveis.append(relatorio["concept_drift"]["nivel"])

        if "critico" in niveis:
            relatorio["nivel_geral"] = "critico"
        elif "alerta" in niveis:
            relatorio["nivel_geral"] = "alerta"
        elif "atencao" in niveis:
            relatorio["nivel_geral"] = "atencao"
        else:
            relatorio["nivel_geral"] = "ok"

        relatorio["drift_detectado"] = relatorio["nivel_geral"] in ("critico", "alerta")
        return relatorio


def gerar_dados_drift(n_samples: int = 5000, shift: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Gera dados sinteticos com shift opcional para teste de drift."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "renda_mensal": rng.exponential(6000, n_samples) + shift * 2000,
        "idade": rng.integers(18, 80, n_samples),
        "score_bureau": np.clip(rng.normal(600, 90, n_samples) + shift * 50, 0, 1000),
        "saldo_atual": rng.exponential(12000, n_samples) + shift * 3000,
        "atrasos_hist": np.clip(rng.poisson(2, n_samples) + int(shift * 3), 0, 30),
        "total_dividas": rng.exponential(18000, n_samples) + shift * 5000,
    })


def main():
    parser = argparse.ArgumentParser(description="Detector de drift")
    parser.add_argument("--config", default="../ops/config/config.yaml")
    parser.add_argument("--reference-data", help="CSV/Parquet dos dados de referencia")
    parser.add_argument("--current-data", help="CSV/Parquet dos dados atuais")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--shift", type=float, default=0.0, help="Shift para dados sinteticos")
    parser.add_argument("--output", help="Path para salvar relatorio JSON")
    args = parser.parse_args()

    if args.reference_data:
        df_ref = pd.read_csv(args.reference_data) if args.reference_data.endswith(".csv") else pd.read_parquet(args.reference_data)
    else:
        df_ref = gerar_dados_drift(args.n_samples)

    if args.current_data:
        df_cur = pd.read_csv(args.current_data) if args.current_data.endswith(".csv") else pd.read_parquet(args.current_data)
    else:
        df_cur = gerar_dados_drift(args.n_samples, shift=args.shift)

    detector = DriftDetector()
    relatorio = detector.gerar_relatorio_completo(df_ref, df_cur)
    logger.info("Drift detectado: %s (nivel=%s)", relatorio["drift_detectado"], relatorio["nivel_geral"])

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(relatorio, f, indent=2, default=str)
        logger.info("Relatorio salvo em %s", args.output)
    else:
        print(json.dumps(relatorio, indent=2, default=str))


if __name__ == "__main__":
    main()
