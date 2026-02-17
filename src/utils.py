import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

#Guardar DataFrame en un archivo CSV
def save_dataframe(df: pd.DataFrame, filepath: Path, index: bool = False):
    df.to_csv(filepath, index=index)
    logger.info(f"DataFrame guardado en {filepath}")

#Cargamos el archivo desde un CSV a un DataFrame
def load_dataframe(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame cargado desde {filepath}")
    return df

#Creamos el timestamp para nombrar los archivos de salida
def create_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Guardamos un diccionario en un archivo JSON
def save_dict_to_json(data: dict, filepath: Path):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Diccionario guardado en {filepath}")

#Configuramos el logging del proyecto
def setup_logging(log_file: Path=None):
    log_format = '%(asctime)s -%(name)s - %(levelname)s - %(message)s'

    if log_file:
        logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ])
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)