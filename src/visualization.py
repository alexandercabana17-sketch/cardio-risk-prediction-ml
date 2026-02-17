import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

#Configuracion del estilo del grafico
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

#Visualizacion de las matrices de confusion para cada modelo
def plot_confusion_matrices(models_dict:dict, X_test, y_test, save_path: Path = None):
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 4, figsize=(20,10))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Confusion matrices saved to {save_path}')
    plt.show()

#Visualizacion de la curva ROC de cada modelo
def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: Path = None):
    plt.figure(figsize=(10, 8))
    for name, model in models_dict.items():
        try:
            y_prob_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        except:
            logger.warning(f"{name} no soporta preddict_proba")

    plt.plot([0, 1], [0, 1], 'k--',linewidth=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Grafico guardado en {save_path}')
    plt.show()

#Visualizacion de la importancia de las features
def plot_feature_importance(model, feature_names, top_n=15, save_path: Path = None):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f'Grafico guardado en {save_path}')
        plt.show()
    else:
        logger.warning('El modelo no tiene el atributo feature_importances_')


#Comparacion de las metricas de rendimiento de los modelos
def plot_models_comparison(results_df: pd.DataFrame, save_path: Path = None):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        data = results_df.sort_values(metric, ascending=False)
        axes[idx].barh(data['model'], data[metric], color = 'coral')
        axes[idx].set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(axis='x', alpha=0.3)

        for i, v in enumerate(data[metric]):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va = 'center')
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f'Grafico guardado en {save_path}')
    plt.show()