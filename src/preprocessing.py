import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

#Elimina los outliers de una columna
def remove_outliers (data: pd.DataFrame, column:str, method:str = 'iqr', lower: float = None, upper: float = None) -> pd.DataFrame:
    data_clean = data.copy()
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif method == 'range':
        if lower is not None and upper is not None:
            data_clean = data_clean[(data_clean[column] >= lower) & (data_clean[column] <= upper)]
    
    removed = len(data) - len(data_clean)
    logger.info(f"Outliers removidos en '{column}': {removed}")
    return data_clean

#Escala las características numéricas utilizando StandardScaler
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, column: list) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[column] = scaler.fit_transform(X_train[column])
    X_test_scaled[column] = scaler.transform(X_test[column])
    
    logger.info(f"Características escaladas: {column}")
    return X_train_scaled, X_test_scaled, scaler

#Codifica las variables categóricas utilizando one-hot encoding o label encoding
def encode_categorical(data:pd.DataFrame, columns:list, method:str = 'onehot') -> pd.DataFrame:
    data_encoded = data.copy()

    if method == 'onehot':
        data_encoded = pd.get_dummies(data_encoded, columns=columns, drop_first=True)
        logger.info(f"One-hot encoding aplicado a {len(columns)} columnas")
    elif method == 'label':
        le = LabelEncoder()
        for col in columns:
            data_encoded[col] = le.fit_transform(data_encoded[col])
        logger.info(f"Label encoding aplicado a {len(columns)} columnas")
    return data_encoded

#Manejamos los valores faltantes utilizando diferentes estrategias (eliminar, imputar con media, mediana o moda)
def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    data_clean = df.copy()
    
    if strategy == 'drop':
        data_clean = data_clean.dropna()
    elif strategy == 'mean':
        data_clean = data_clean.fillna(data_clean.mean())
    elif strategy == 'median':
        data_clean = data_clean.fillna(data_clean.median())
    elif strategy == 'mode':
        data_clean = data_clean.fillna(data_clean.mode().iloc[0])
    
    logger.info(f"Valores faltantes manejados con estrategia: {strategy}")
    return data_clean



