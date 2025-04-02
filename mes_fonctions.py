import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import roc_curve

def create_and_evaluate_model(df, target_column, model_type='GradientBoosting', random_state=42):
    # Séparation des features et du target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Imputation des valeurs manquantes sans modifier le DataFrame original
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Encoder les labels si nécessaire
    if y.dtypes == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=random_state)
    
    # Choix du modèle
    if model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(random_state=random_state)
    elif model_type == 'LightGBM':
        model = lgb.LGBMClassifier(random_state=random_state)
    elif model_type == 'CatBoost':
        model = cb.CatBoostClassifier(verbose=0, random_state=random_state)
    else:
        raise ValueError("Modèle non supporté. Choisissez parmi 'GradientBoosting', 'XGBoost', 'LightGBM' ou 'CatBoost'.")

    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions sur le test set
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Validation croisée
    cross_val = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy')
    
    # Affichage des résultats
    print(f"Modèle: {model_type}")
    print(f"Exactitude: {accuracy:.4f}")
    print(f"Précision: {precision:.4f}")
    print(f"Rappel: {recall:.4f}")
    print(f"F-mesure: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"Validation croisée (accuracy): {cross_val.mean():.4f} ± {cross_val.std():.4f}")
    
    # Affichage de la matrice de confusion et de la courbe AUC-ROC
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Matrice de confusion
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=axes[0])
    axes[0].set_title("Matrice de Confusion")
    axes[0].set_xlabel("Prédictions")
    axes[0].set_ylabel("Vraies étiquettes")
    
    # Courbe AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    axes[1].plot(fpr, tpr, color='blue', lw=2)
    axes[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    axes[1].set_title("Courbe AUC-ROC")
    axes[1].set_xlabel("Taux de faux positifs")
    axes[1].set_ylabel("Taux de vrais positifs")
    
    plt.show()

# Exemple d'appel de la fonction
# create_and_evaluate_model(df, 'target', model_type='XGBoost')
