# Predicción de Enfermedades Cardiovasculares mediante Machine Learning

Sistema de clasificación supervisada para predecir la presencia de enfermedades cardiovasculares utilizando datos clínicos y de estilo de vida de pacientes. El proyecto implementa y compara siete algoritmos de aprendizaje automático, logrando un AUC de 72.1% con Gradient Boosting.

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación](#instalación)
- [Uso](#uso)
- [Modelos Implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Metodología](#metodología)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Descripción del Proyecto

Este proyecto desarrolla un sistema de predicción de enfermedades cardiovasculares mediante técnicas de Machine Learning. El objetivo principal es clasificar pacientes según su riesgo de padecer enfermedades cardíacas basándose en variables clínicas y demográficas.

### Objetivos

- Implementar un pipeline completo de Machine Learning para predicción médica
- Comparar el rendimiento de múltiples algoritmos de clasificación
- Identificar los factores de riesgo más relevantes mediante análisis de importancia de características
- Proporcionar un sistema reproducible y escalable

## Dataset

**Características del Dataset:**
- Número de registros: 70,000 pacientes
- Número de variables: 12 features + 1 variable objetivo
- Tipo de problema: Clasificación binaria (presencia/ausencia de enfermedad cardiovascular)
- Origen: Dataset público de investigación cardiovascular

### Variables

**Variables Demográficas:**
- Edad (años)
- Género
- Altura (cm)
- Peso (kg)

**Variables Clínicas:**
- Presión arterial sistólica (mmHg)
- Presión arterial diastólica (mmHg)
- Nivel de colesterol (categórico: normal, alto, muy alto)
- Nivel de glucosa (categórico: normal, alto, muy alto)

**Variables de Estilo de Vida:**
- Fumador (binario)
- Consumo de alcohol (binario)
- Actividad física (binario)

**Variable Objetivo:**
- Presencia de enfermedad cardiovascular (0 = No, 1 = Sí)

## Estructura del Proyecto
```
cardio-risk-prediction-ml/
│
├── data/
│   ├── raw/                    # Datos originales
│   └── processed/              # Datos procesados
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuraciones globales
│   ├── data_loader.py          # Carga y validación de datos
│   ├── preprocessing.py        # Preprocesamiento de datos
│   ├── feature_engineering.py  # Ingeniería de características
│   ├── models.py               # Implementación de modelos
│   ├── evaluation.py           # Métricas de evaluación
│   ├── visualization.py        # Visualizaciones
│   └── utils.py                # Funciones utilitarias
│
├── notebooks/
│   └── analisis_exploratorio.ipynb  # Análisis exploratorio de datos
│
├── models/                     # Modelos entrenados
├── results/
│   ├── figures/                # Gráficos y visualizaciones
│   └── reports/                # Reportes de métricas
│
├── tests/                      # Tests unitarios
│   └── test_preprocessing.py
│
├── main.py                     # Script principal del pipeline
├── requirements.txt            # Dependencias del proyecto
├── .gitignore
└── README.md
```

## Tecnologías Utilizadas

**Lenguaje de Programación:**
- Python 3.8+

**Bibliotecas Principales:**
- **Procesamiento de Datos:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualización:** matplotlib, seaborn
- **Análisis Exploratorio:** jupyter, notebook

**Control de Versiones:**
- Git, GitHub

## Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/alexandercabana17-sketch/cardio-risk-prediction-ml.git
cd cardio-risk-prediction-ml
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecución del Pipeline Completo

Para ejecutar el pipeline completo de entrenamiento y evaluación:
```bash
python main.py
```

Este comando realizará:
1. Carga de datos preprocesados
2. Ingeniería de características
3. Preparación de datos (escalado, división train-test)
4. Entrenamiento de siete modelos
5. Evaluación y comparación
6. Generación de visualizaciones
7. Guardado del mejor modelo

### Análisis Exploratorio

Para revisar el análisis exploratorio de datos:
```bash
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

### Resultados Generados

Después de la ejecución, se generarán:
- **Modelos entrenados:** `models/best_model_*.pkl`
- **Scaler:** `models/scaler_*.pkl`
- **Reporte de métricas:** `results/model_comparison_*.csv`
- **Visualizaciones:**
  - Matrices de confusión
  - Curvas ROC
  - Comparación de modelos
  - Importancia de características

## Modelos Implementados

El proyecto implementa y compara los siguientes algoritmos de clasificación:

| Modelo | Tipo | Características |
|--------|------|-----------------|
| Logistic Regression | Lineal | Baseline simple y rápido |
| Decision Tree | Árbol | Interpretable, captura no linealidades |
| Random Forest | Ensemble | Robusto, reduce overfitting |
| Gradient Boosting | Ensemble | Alta precisión, construcción secuencial |
| XGBoost | Ensemble | Optimizado, regularización avanzada |
| Support Vector Machine | Kernel | Efectivo en espacios de alta dimensión |
| K-Nearest Neighbors | Basado en instancias | Simple, no paramétrico |

## Resultados

### Rendimiento de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Gradient Boosting | 0.7150 | 0.5786 | 0.2767 | 0.3748 | 0.7210 |
| XGBoost | 0.7139 | 0.5787 | 0.2764 | 0.3733 | 0.7180 |
| Random Forest | 0.7133 | 0.5632 | 0.2595 | 0.3546 | 0.7070 |
| Logistic Regression | 0.6748 | 0.5628 | 0.2011 | 0.2963 | 0.6900 |
| Decision Tree | 0.7076 | 0.5377 | 0.2562 | 0.3470 | 0.6860 |
| SVM | 0.6722 | 0.6368 | 0.1313 | 0.2178 | 0.6680 |
| KNN | 0.6690 | 0.4524 | 0.2864 | 0.3508 | 0.6230 |

### Mejor Modelo

**Gradient Boosting** obtuvo el mejor rendimiento general con:
- AUC-ROC: 72.1%
- Accuracy: 71.5%
- Mejor balance entre sensibilidad y especificidad

### Características Más Importantes

Las variables con mayor importancia predictiva son:
1. Edad
2. Presión arterial sistólica
3. Índice de masa corporal (BMI)
4. Presión arterial media
5. Nivel de colesterol

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones
- Detección de valores atípicos
- Análisis de correlaciones
- Visualización de patrones

### 2. Ingeniería de Características
Se crearon las siguientes características derivadas:
- **age_years:** Edad convertida de días a años
- **bmi:** Índice de Masa Corporal (peso/altura²)
- **pulse_pressure:** Presión de pulso (sistólica - diastólica)
- **map:** Presión Arterial Media [(sistólica + 2×diastólica) / 3]
- **bmi_category:** Categorías de BMI (underweight, normal, overweight, obese)

### 3. Preprocesamiento
- Eliminación de valores imposibles
- Escalado de características numéricas (StandardScaler)
- Codificación de variables categóricas (One-Hot Encoding)
- División estratificada train-test (80-20)

### 4. Entrenamiento y Evaluación
- Entrenamiento de múltiples modelos
- Validación cruzada
- Métricas múltiples (Accuracy, Precision, Recall, F1, AUC-ROC)
- Análisis de matrices de confusión
- Curvas ROC comparativas

### 5. Interpretabilidad
- Análisis de importancia de características
- Visualización de resultados
- Documentación de hallazgos
