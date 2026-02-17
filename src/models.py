from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

#Definimos la clase ModelTrainer para entrenar y evaluar diferentes modelos de clasificación
class ModelTrainer:
    #Inicializamos el entrenador de modelos
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
    
    #Configuramos los modelos a entrenar con sus hiperparámetros
    def initialize_models(self, models_config: dict):
        self.models = {
            'Logistic Regression': LogisticRegression(**models_config['Logistic Regression']),
            'Decision Tree': DecisionTreeClassifier(**models_config['Decision Tree']),
            'Random Forest': RandomForestClassifier(**models_config['Random Forest']),
            'Gradient Boosting': GradientBoostingClassifier(**models_config['Gradient Boosting']),
            'SVM': SVC(**models_config['SVM']),
            'KNN': KNeighborsClassifier(**models_config['KNN']),
            'XGBoost': XGBClassifier(**models_config['XGBoost'])
        }
        logger.info(f"Modelos inicializados: {list(self.models.keys())}")
    
    #Entrenamos un modelo específico con los datos de entrenamiento
    def train_model(self, name:str, X_train, y_train):
        logger.info(f"Entrenando modelo: {name}")

        model = self.models[name]
        model.fit(X_train, y_train)
        self.trained_models[name] = model

        logger.info(f"Modelo '{name}' entrenado exitosamente")
    
    #Entrenamos todos los modelos definidos en la configuración
    def train_all_models(self, X_train, y_train):
        for name in self.models.keys():
            self.train_model(name, X_train, y_train)
        
        logger.info("Todos los modelos han sido entrenados")
    
    #Realizamos validacion cruzada
    def cross_validate(self, name:str, X, y, cv:int = 5):
        model = self.models[name]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        logger.info(f"Cross-validation scores para '{name}': {scores}")
        return {'mean': scores.mean(), 'std': scores.std(), 'scores': scores}
    
    #Guardamos el modelo entrenado en un archivo utilizando joblib
    def save_model(self, name:str, filepath: Path):
        if name in self.trained_models:
            joblib.dump(self.trained_models[name], filepath)
            logger.info(f"Modelo '{name}' guardado en {filepath}")
        else:
            logger.warning(f"Modelo '{name}' no encontrado. No se ha guardado.")
    
    def load_model(self, name:str, filepath: Path):
        self.trained_models[name] = joblib.load(filepath)
        logger.info(f"Modelo '{name}' cargado desde {filepath}")



       