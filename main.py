import sys
from pathlib import Path
import traceback

import joblib
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path(__file__).parent/ "src"))

import pandas as pd
import logging
from src.config import *
from src.feature_engineering import engineer_features
from src.preprocessing import scale_features
from src.models import ModelTrainer
from src.evaluation import compare_models
from src.visualization import(plot_confusion_matrices, plot_roc_curves, plot_models_comparison, plot_feature_importance)
from src.utils import create_timestamp, save_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info('Cargando datos')
        if not CLEAN_DATA_FILE.exists():
            raise FileNotFoundError(f"Archivo de datos limpio no encontrado: {CLEAN_DATA_FILE}")
        
        data = pd.read_csv(CLEAN_DATA_FILE)
        print(f'Datos cargados con exito. Shape: {data.shape} - Columnas: {list(data.columns)}')

        if TARGER_COLUMN not in data.columns:
            raise ValueError(f"Columna objetivo '{TARGER_COLUMN}' no encontrada en los datos.")
        
        logger.info('\nAplicando feature engineering')

        data_original_shape = data.shape[1]
        data = engineer_features(data)
        print('Feature engineering completado')

        logger.info('\nPreprocesando datos para el modelado')

        X = data.drop(columns=[TARGER_COLUMN])
        y = data[TARGER_COLUMN]

        columns_to_drop = []
        if 'id' in X.columns:
            columns_to_drop.append('id')
        
        if columns_to_drop:
            X = X.drop(columns=columns_to_drop)
            print(f"Columnas eliminadas: {columns_to_drop}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RANDOM_STATE, stratify = y)
        print(f"Datos divididos en train y test. Train shape: {X_train.shape} - Test shape: {X_test.shape}")

        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, numerical_cols)
        print(f"Features escalados. Columnas escaladas: {numerical_cols}")

        logger.info('\nEntrenando modelos')
        print(f'\nModelos a entrenar: {len(MODELS_CONFIG)}')
        trainer = ModelTrainer(RANDOM_STATE)
        trainer.initialize_models(MODELS_CONFIG)

        print(f'Entrenando modelos...')
        for idx, name in enumerate(trainer.models.keys(),1):
            print(f'[{idx}/{len(trainer.models.keys())}] Entrenando modelo: {name}')
            trainer.train_model(name, X_train_scaled, y_train)
        
        print(f'Modelos entrenados con exito')

        logger.info('\nComparando modelos')
        results_df = compare_models(trainer.models, X_test_scaled, y_test)
        print(f'Modelos comparados con exito. Resultados:')
        print(results_df.to_string(index=False))

        #Guardamos los resultados
        timestamp = create_timestamp()
        results_file = RESULTS_DIR / f'model_comparison_{timestamp}.csv'
        save_dataframe(results_df, results_file)
        print(f'Resultados guardados en: {results_file}')

        logger.info('\nGenerando visualizaciones')
        
        #Matrices de confusion
        plot_confusion_matrices(trainer.trained_models, X_test_scaled, y_test, save_path=FIGURES_DIR / f'confusion_matrices_{timestamp}.png')

        #Curvas ROC
        plot_roc_curves(trainer.trained_models, X_test_scaled, y_test, save_path=FIGURES_DIR / f'roc_curves_{timestamp}.png')

        #Comparacion de modelos
        plot_models_comparison(results_df, save_path=FIGURES_DIR / f'model_comparison_{timestamp}.png')

        #Importancia de features
        best_model_name = results_df.iloc[0]['model']
        best_model = trainer.trained_models[best_model_name]

        if hasattr(best_model, 'feature_importances_'):
            plot_feature_importance(best_model, X_train.columns, save_path=FIGURES_DIR / f'feature_importance_{best_model_name}_{timestamp}.png')
        else:
            print(f"El modelo '{best_model_name}' no tiene atributo 'feature_importances_' para mostrar la importancia de las features.")

        print(f'Visualizaciones generadas y guardadas en: {FIGURES_DIR}')

        logger.info('\nGuardando el mejor modelo...')

        #Guardamos el modelo
        model_file = MODELS_DIR / f'best_model_{best_model_name.replace(" ", "_")}_{timestamp}.pkl'
        trainer.save_model(best_model_name, model_file)

        #Guardamos el scaler
        scaler_file = MODELS_DIR / f'scaler_{timestamp}.plk'
        joblib.dump(scaler, scaler_file)

        print(f'Mejor modelo guardado en: {model_file}')
        print(f'Scaler guardado en: {scaler_file}')

        logger.info('Proceso completado con exito')

    except Exception as e:
        logger.error(f"Ocurrio un error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()






