import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

#Convertimos la edad de dias a años
def create_age_years(data:pd.DataFrame, age_column: str = 'edad') ->pd.DatraFrame:
    data_new = data.copy()
    if age_column in data_new.columns:
        data_new['edad_años'] = (data_new[age_column] / 365).astype(int)
        logger.info(f"Feature 'edad_años' creada")
    else:
        logger.warning(f"Columna '{age_column}' no encontrada.")
    
    return data_new

#Calculamos el indice de masa corporal (BMI) a partir del peso y la altura
def create_bmi(data: pd.DataFrame, weight_col: str = 'peso', height_col: str = 'altura')-> pd.DataFrame:
    data_new = data.copy()
    if weight_col in data_new.columns and height_col in data_new.columns:
        data_new['bmi'] = data_new[weight_col] / ((data_new[height_col] / 100) ** 2)
        logger.info(f"Feature 'bmi' creada")
    else:
        logger.warning(f"Columnas '{weight_col}' o '{height_col}' no encontradas.")
    
    return data_new

#Calculamos la presion del pulso a partir de la presion arterial sistolica y diastolica
def create_pulse_pressure(data: pd.DataFrame, systolic_col: str = 'presion_arterial_sistolica', diastolic_col: str = 'presion_arterial_diastolica') -> pd.DataFrame:
    data_new = data.copy()
    if systolic_col in data_new.columns and diastolic_col in data_new.columns:
        data_new['pulse_pressure'] = data_new[systolic_col] - data_new[diastolic_col]
        logger.info(f"Feature 'pulse_pressure' creada")
    else:
        logger.warning(f"Columnas '{systolic_col}' o '{diastolic_col}' no encontradas.")
    
    return data_new

#Calculamos la presion arterial media (MAP) a partir de la presion arterial sistolica y diastolica
def create_map(df: pd.DataFrame, systolic_col: str = 'presion_arterial_sistolica', diastolic_col: str = 'presion_arterial_diastolica') -> pd.DataFrame:
    data_new = df.copy()
    if systolic_col in data_new.columns and diastolic_col in data_new.columns:
        data_new['presion_arterial_media'] = (data_new[systolic_col] + 2 * data_new[diastolic_col]) / 3
        logger.info(f"Feature 'presion_arterial_media' creada")
    else:
        logger.warning(f"Columnas '{systolic_col}' o '{diastolic_col}' no encontradas.")
    
    return data_new

#Creamos categorias de BMI (bajo peso, normal, sobrepeso, obesidad) a partir del indice de masa corporal (BMI)
def create_bmi_category(data: pd. DataFrame, bmi_col: str = 'bmi') -> pd.DataFrame:
    data_new = data.copy()
    if bmi_col in data_new.columns:
        data_new['bmi_category'] = pd.cut(data_new[bmi_col], bins = [0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        logger.info(f"Feature 'bmi_category' creada")

        #Aplicamos el onehot encoder
        bmi_dumies = pd.get_dummies(data_new['bmi_category'], drop_first=True)
        data_new = pd.concat([data_new, bmi_dumies], axis=1)

        #Eliminamos la columna original
        data_new = data_new.drop(columns=['bmi_category'])
        logger.info(f'One hot ecoder aplicado: {list(bmi_dumies.columns)}')
    else:
        logger.warning(f"Columna '{bmi_col}' no encontrada.")
    
    return data_new

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data_new = data.copy()
    data_new = create_age_years(data_new)
    data_new = create_bmi(data_new)
    data_new = create_pulse_pressure(data_new)
    data_new = create_map(data_new)
    data_new = create_bmi_category(data_new)
    
    features_created = data_new.shape[1] - data.shape[1]
    logger.info(f"Nuevas features creadfas: {features_created}")

    return data_new