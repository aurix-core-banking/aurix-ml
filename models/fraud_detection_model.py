"""
AUREUS Machine Learning - Modelo de Detecção de Fraude
Sistema de ML para detecção de transações fraudulentas em tempo real
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Modelo de detecção de fraude para transações bancárias"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para o modelo"""
        df = df.copy()
        
        # Converter data_transacao para datetime
        df['data_transacao'] = pd.to_datetime(df['data_transacao'])
        
        # Features temporais
        df['hour'] = df['data_transacao'].dt.hour
        df['day_of_week'] = df['data_transacao'].dt.dayofweek
        df['day_of_month'] = df['data_transacao'].dt.day
        df['month'] = df['data_transacao'].dt.month
        
        # Features de valor
        df['valor_log'] = np.log1p(df['valor'])
        df['valor_sqrt'] = np.sqrt(df['valor'])
        df['is_high_value'] = (df['valor'] > df['valor'].quantile(0.95)).astype(int)
        df['is_round_value'] = (df['valor'] % 1000 == 0).astype(int)
        
        # Features de localização
        df['is_same_city'] = (df['cidade'] == df['cidade'].mode()[0]).astype(int)
        df['is_same_state'] = (df['estado'] == df['estado'].mode()[0]).astype(int)
        
        # Features de canal
        df['is_mobile'] = (df['canal'] == 'MOBILE').astype(int)
        df['is_web'] = (df['canal'] == 'WEB').astype(int)
        df['is_atm'] = (df['canal'] == 'ATM').astype(int)
        
        # Features de horário
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Features de dispositivo
        df['is_iphone'] = df['dispositivo'].str.contains('iPhone', na=False).astype(int)
        df['is_android'] = df['dispositivo'].str.contains('Android', na=False).astype(int)
        df['is_samsung'] = df['dispositivo'].str.contains('Samsung', na=False).astype(int)
        
        # Features de IP (simplificado)
        df['ip_first_octet'] = df['ip_address'].str.split('.').str[0].astype(float)
        df['ip_second_octet'] = df['ip_address'].str.split('.').str[1].astype(float)
        
        # Features de user agent
        df['is_chrome'] = df['user_agent'].str.contains('Chrome', na=False).astype(int)
        df['is_firefox'] = df['user_agent'].str.contains('Firefox', na=False).astype(int)
        df['is_safari'] = df['user_agent'].str.contains('Safari', na=False).astype(int)
        
        # Features de score de risco
        df['score_risco_normalized'] = df['score_risco'] / df['score_risco'].max()
        
        # Features de frequência (simuladas)
        df['transactions_last_hour'] = np.random.randint(0, 10, len(df))
        df['transactions_last_day'] = np.random.randint(0, 50, len(df))
        df['avg_transaction_value'] = df.groupby('conta_id')['valor'].transform('mean')
        df['transaction_value_ratio'] = df['valor'] / df['avg_transaction_value']
        
        # Selecionar features numéricas
        numeric_features = [
            'valor', 'valor_log', 'valor_sqrt', 'hour', 'day_of_week', 'day_of_month', 'month',
            'is_high_value', 'is_round_value', 'is_same_city', 'is_same_state',
            'is_mobile', 'is_web', 'is_atm', 'is_business_hours', 'is_night', 'is_weekend',
            'is_iphone', 'is_android', 'is_samsung', 'ip_first_octet', 'ip_second_octet',
            'is_chrome', 'is_firefox', 'is_safari', 'score_risco_normalized',
            'transactions_last_hour', 'transactions_last_day', 'transaction_value_ratio'
        ]
        
        # Codificar features categóricas
        categorical_features = ['tipo_transacao', 'status', 'cidade', 'estado']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                numeric_features.append(f'{feature}_encoded')
        
        # Selecionar features finais
        self.feature_columns = [col for col in numeric_features if col in df.columns]
        df_features = df[self.feature_columns].fillna(0)
        
        return df_features
    
    def train(self, df: pd.DataFrame, target_column: str = 'is_fraud'):
        """Treina o modelo de detecção de fraude"""
        # Reproductibilidade das features sintéticas (transactions_last_hour, etc.)
        np.random.seed(42)
        print("Preparando dados para treinamento...")
        
        # Preparar features
        X = self.prepare_features(df)
        
        # Criar target se não existir
        if target_column not in df.columns:
            # Simular target baseado em regras de negócio
            df[target_column] = (
                (df['valor'] > df['valor'].quantile(0.99)) |
                (df['hour'] < 6) |
                (df['hour'] > 22) |
                (df['day_of_week'] >= 5) |
                (df['score_risco'] > 0.8)
            ).astype(int)
        
        y = df[target_column]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Treinando modelo Isolation Forest...")
        # Treinar Isolation Forest
        self.isolation_forest.fit(X_train_scaled)
        
        print("Treinando modelo Random Forest...")
        # Treinar Random Forest
        self.random_forest.fit(X_train_scaled, y_train)
        
        # Avaliar modelos
        self._evaluate_models(X_test_scaled, y_test)
        
        self.is_trained = True
        print("Modelo treinado com sucesso!")
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: pd.Series):
        """Avalia os modelos treinados"""
        print("\n=== Avaliação dos Modelos ===")
        
        # Isolation Forest
        iso_predictions = self.isolation_forest.predict(X_test)
        iso_scores = self.isolation_forest.score_samples(X_test)
        
        print("Isolation Forest:")
        print(f"Anomalias detectadas: {np.sum(iso_predictions == -1)}")
        print(f"Score médio: {np.mean(iso_scores):.4f}")
        
        # Random Forest
        rf_predictions = self.random_forest.predict(X_test)
        rf_probabilities = self.random_forest.predict_proba(X_test)[:, 1]
        
        print("\nRandom Forest:")
        print(classification_report(y_test, rf_predictions))
        print(f"AUC Score: {roc_auc_score(y_test, rf_probabilities):.4f}")
        
        # Matriz de confusão
        print("\nMatriz de Confusão (Random Forest):")
        print(confusion_matrix(y_test, rf_predictions))
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Faz predições de fraude"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        # Preparar features
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Predições
        iso_predictions = self.isolation_forest.predict(X_scaled)
        iso_scores = self.isolation_forest.score_samples(X_scaled)
        rf_predictions = self.random_forest.predict(X_scaled)
        rf_probabilities = self.random_forest.predict_proba(X_scaled)[:, 1]
        
        # Combinar predições
        combined_scores = (iso_scores + rf_probabilities) / 2
        combined_predictions = ((iso_predictions == -1) | (rf_predictions == 1)).astype(int)
        
        return {
            'isolation_forest': {
                'predictions': iso_predictions.tolist(),
                'scores': iso_scores.tolist()
            },
            'random_forest': {
                'predictions': rf_predictions.tolist(),
                'probabilities': rf_probabilities.tolist()
            },
            'combined': {
                'predictions': combined_predictions.tolist(),
                'scores': combined_scores.tolist()
            }
        }
    
    def save_model(self, filepath: str):
        """Salva o modelo treinado"""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'random_forest': self.random_forest,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega um modelo treinado"""
        model_data = joblib.load(filepath)
        self.isolation_forest = model_data['isolation_forest']
        self.random_forest = model_data['random_forest']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"Modelo carregado de: {filepath}")

class CreditScoringModel:
    """Modelo de scoring de crédito"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=15
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara features para scoring de crédito"""
        df = df.copy()
        
        # Features de renda
        df['renda_log'] = np.log1p(df['renda_mensal'])
        df['renda_per_capita'] = df['renda_mensal'] / (df['pessoas_residencia'] + 1)
        
        # Features de idade
        df['idade_squared'] = df['idade'] ** 2
        df['is_young'] = (df['idade'] < 30).astype(int)
        df['is_middle_age'] = ((df['idade'] >= 30) & (df['idade'] <= 50)).astype(int)
        df['is_senior'] = (df['idade'] > 50).astype(int)
        
        # Features de escolaridade
        df['escolaridade_numeric'] = df['escolaridade'].map({
            'FUNDAMENTAL': 1,
            'MEDIO': 2,
            'SUPERIOR': 3,
            'POS_GRADUACAO': 4,
            'MESTRADO': 5,
            'DOUTORADO': 6
        }).fillna(0)
        
        # Features de estado civil
        df['is_married'] = (df['estado_civil'] == 'CASADO').astype(int)
        df['is_single'] = (df['estado_civil'] == 'SOLTEIRO').astype(int)
        
        # Features de localização
        df['is_capital'] = df['cidade'].isin(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Brasília']).astype(int)
        
        # Features de profissão
        df['is_high_income_profession'] = df['profissao'].isin([
            'MEDICO', 'ENGENHEIRO', 'ADVOGADO', 'DENTISTA', 'ARQUITETO'
        ]).astype(int)
        
        # Features de conta
        df['tempo_conta_meses'] = (datetime.now() - pd.to_datetime(df['data_abertura'])).dt.days / 30
        df['saldo_renda_ratio'] = df['saldo_atual'] / (df['renda_mensal'] + 1)
        
        # Features de transações
        df['transacoes_mes'] = df['transacoes_ultimo_mes']
        df['valor_medio_transacao'] = df['valor_total_transacoes'] / (df['transacoes_mes'] + 1)
        
        # Selecionar features
        feature_columns = [
            'renda_mensal', 'renda_log', 'renda_per_capita', 'idade', 'idade_squared',
            'is_young', 'is_middle_age', 'is_senior', 'escolaridade_numeric',
            'is_married', 'is_single', 'is_capital', 'is_high_income_profession',
            'tempo_conta_meses', 'saldo_renda_ratio', 'transacoes_mes',
            'valor_medio_transacao', 'pessoas_residencia'
        ]
        
        return df[feature_columns].fillna(0)
    
    def train(self, df: pd.DataFrame, target_column: str = 'score_credito'):
        """Treina o modelo de scoring de crédito"""
        print("Treinando modelo de scoring de crédito...")
        
        X = self.prepare_features(df)
        y = df[target_column]
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelo
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        print("Modelo de scoring de crédito treinado!")
    
    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Prediz score de crédito"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Gera dados de exemplo para treinamento"""
    np.random.seed(42)
    
    data = {
        'id': range(1, n_samples + 1),
        'conta_id': np.random.randint(1, 100, n_samples),
        'tipo_transacao': np.random.choice(['PIX', 'TED', 'DOC', 'SAQUE', 'DEPOSITO'], n_samples),
        'valor': np.random.exponential(500, n_samples),
        'data_transacao': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'status': np.random.choice(['APROVADA', 'REJEITADA', 'PENDENTE'], n_samples, p=[0.8, 0.15, 0.05]),
        'canal': np.random.choice(['MOBILE', 'WEB', 'ATM', 'AGENCIA'], n_samples, p=[0.6, 0.3, 0.08, 0.02]),
        'dispositivo': np.random.choice(['iPhone 14', 'Samsung Galaxy', 'Chrome', 'Firefox'], n_samples),
        'ip_address': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'user_agent': np.random.choice(['Mozilla/5.0 Chrome', 'Mozilla/5.0 Firefox', 'Mozilla/5.0 Safari'], n_samples),
        'latitude': np.random.uniform(-33.0, 5.0, n_samples),
        'longitude': np.random.uniform(-74.0, -34.0, n_samples),
        'cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Brasília'], n_samples),
        'estado': np.random.choice(['SP', 'RJ', 'MG', 'BA', 'DF'], n_samples),
        'pais': ['Brasil'] * n_samples,
        'score_risco': np.random.uniform(0, 1, n_samples),
        'aprovada': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'tempo_processamento_ms': np.random.randint(50, 500, n_samples),
        'renda_mensal': np.random.exponential(5000, n_samples),
        'idade': np.random.randint(18, 80, n_samples),
        'escolaridade': np.random.choice(['FUNDAMENTAL', 'MEDIO', 'SUPERIOR', 'POS_GRADUACAO'], n_samples, p=[0.1, 0.3, 0.5, 0.1]),
        'estado_civil': np.random.choice(['SOLTEIRO', 'CASADO', 'DIVORCIADO', 'VIUVO'], n_samples, p=[0.4, 0.4, 0.15, 0.05]),
        'profissao': np.random.choice(['ENGENHEIRO', 'MEDICO', 'ADVOGADO', 'COMERCIANTE', 'FUNCIONARIO'], n_samples),
        'pessoas_residencia': np.random.randint(1, 6, n_samples),
        'data_abertura': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'saldo_atual': np.random.exponential(2000, n_samples),
        'transacoes_ultimo_mes': np.random.randint(0, 100, n_samples),
        'valor_total_transacoes': np.random.exponential(10000, n_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """Função principal para treinar e testar os modelos"""
    np.random.seed(42)
    print("=== AUREUS Machine Learning - Detecção de Fraude ===")
    
    # Gerar dados de exemplo
    print("Gerando dados de exemplo...")
    df = generate_sample_data(5000)
    
    # Treinar modelo de detecção de fraude
    print("\n1. Treinando modelo de detecção de fraude...")
    fraud_model = FraudDetectionModel()
    fraud_model.train(df)
    
    # Salvar modelo
    fraud_model.save_model('fraud_detection_model.pkl')
    
    # Treinar modelo de scoring de crédito
    print("\n2. Treinando modelo de scoring de crédito...")
    credit_model = CreditScoringModel()
    credit_model.train(df)
    
    # Testar predições
    print("\n3. Testando predições...")
    test_df = df.head(100)
    fraud_predictions = fraud_model.predict(test_df)
    credit_scores = credit_model.predict_score(test_df)
    
    print(f"Fraudes detectadas: {sum(fraud_predictions['combined']['predictions'])}")
    print(f"Score médio de crédito: {np.mean(credit_scores):.2f}")
    
    print("\n=== Treinamento concluído com sucesso! ===")

if __name__ == "__main__":
    main()