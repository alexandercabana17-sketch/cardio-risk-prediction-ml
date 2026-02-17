import pandas as pd
import numpy as np
from sklearn.metrics import(accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, classification_report)
from typing import Dict
import logging

logger = logging.getLogger(__name__)

#Evaluamos el modelo utilizando las métricas de clasificación
def evaluate_model(y_true, y_pred, y_pred_proba=None) -> Dict:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            pass
    return metrics

#Compramos multiples modelos
def compare_models(models_dict: Dict, X_test, y_test) -> pd.DataFrame:
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = None
        
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        metrics['model'] = name
        results.append(metrics)

        logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('accuracy', ascending=False).reset_index(drop=True)

    return df_results

#Imprimimos el resporte de clasificacion detallado
def print_claification_report(y_true, y_pred, model_name:str):
    print(f"Clasification report - {model_name}")
    print(classification_report(y_true, y_pred))
    print(f"\nMatriz de Confusion:")
    print(confusion_matrix(y_true, y_pred))

#Obtenemos las metricas de la matriz de confusion
def get_confusion_matrix_matrics(y_true, y_pred) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).reavel()
    return{
        'true_negatives': tn,
        'false_positive': fp,
        'false_negaftive': fn,
        'true_positive' : tp,
        'specificity': tn/(tn+fp) if (tn+fp) > 0 else 0,
        'sensitivity': tp/(tp+fn) if (tp+fn) > 0 else 0,
    }

    