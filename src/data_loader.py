import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#Carga los datos desde el archivo CSV
def load_data(filepath: Path, separator: str = ",") -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath, sep=separator)
        logger.info(f"Datos cargados: {data.shape[0]} filas y {data.shape[1]} columnas")
        return data
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar data: {e}")
        raise

#Obtiene información general del dataset
def get_data_info(data: pd.DataFrame) -> dict:
    info = {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "duplicates": data.duplicated().sum(),
        "memory_usage": data.memory_usage(deep=True).sum()/1024**2}
    return info

#Valida que el dataset tenga las columans requeridas para el análisis
def validate_dataset(data: pd.DataFrame, required_columns: list) -> bool:
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        logger.error(f"Columnas faltantes: {missing_cols}")
        return False
    logger.info("Validación exitosa: Todas las columnas requeridas están presentes")
    return True