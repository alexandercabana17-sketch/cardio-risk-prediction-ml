from pathlib import Path

#Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESPORTS_DIR = PROJECT_ROOT / "reports"

#Crear los directorios si es que no existen
for directory in [MODELS_DIR, FIGURES_DIR, RESPORTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

#Archivos de los datos
RAW_DATA_FILE = RAW_DATA_DIR / "enfermedades_cardiacas.csv"
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "enfermedades_cardiacas_limpio.csv"

#Parametros del modelado
RANDOM_STATE = 42
TEST_SIZE = 0.2

#Definimos el target
TARGER_COLUMN = "presencia_enfermedad"

#Cofniguracion de los modelos
MODELS_CONFIG = {'Logistic Regression': {'max_iter': 1000, 'random_state': RANDOM_STATE},
                 'Decision Tree': {'max_depth': 10, 'random_state': RANDOM_STATE},
                 'Random Forest': {'n_estimators': 100, 'max_depth': 15, 'random_state': RANDOM_STATE, 'n_jobs': -1},
                 'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': RANDOM_STATE},
                 'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': RANDOM_STATE, 'eval_metric': 'logloss'},
                 'SVM': {'kernel': 'rbf', 'random_state': RANDOM_STATE, 'probability': True},
                 'KNN': {'n_neighbors': 5, 'n_jobs': -1}}

#Configuracion de visualizaciones
FIGURE_SIZE = (12, 6)
STYLE = 'seaborn-v0_8-darkgrid'
PALETTE = 'husl'
DPI = 300