import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgb
import gc
import time
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
# import xgboost as xgb
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from sklearn.impute import SimpleImputer

def kfold_classification(df, model_type, num_folds=5, stratified=True, show_feature_importance=True):
    start_time = time.time()
    
    train_df = df[df['TARGET'].notnull()]
    print(f"Starting {model_type}. Train shape: {train_df.shape}")
    
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42) if stratified else KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR']]
    
    fpr_all, tpr_all, roc_auc_all, cm_all = [], [], [], []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        
        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(train_x)
        valid_x = imputer.transform(valid_x)
        
        if np.min(np.bincount(train_y)) / np.max(np.bincount(train_y)) < 0.5:
            sm = SMOTE(random_state=42)
            train_x, train_y = sm.fit_resample(train_x, train_y)
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'xgboost': XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42),
            'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, random_state=42)
        }
        
        if model_type not in models:
            raise ValueError("Unsupported model type")
        
        model = models[model_type]
        model.fit(train_x, train_y)
        
        valid_preds_proba = model.predict_proba(valid_x)[:, 1]
        oof_preds[valid_idx] = valid_preds_proba
        
        fpr, tpr, _ = roc_curve(valid_y, valid_preds_proba)
        roc_auc = roc_auc_score(valid_y, valid_preds_proba)
        fpr_all.append(fpr)
        tpr_all.append(tpr)
        roc_auc_all.append(roc_auc)
        
        y_pred = (valid_preds_proba > 0.5).astype(int)
        cm = confusion_matrix(valid_y, y_pred)
        cm_all.append(cm)
        
        accuracy = accuracy_score(valid_y, y_pred)
        precision = precision_score(valid_y, y_pred)
        recall = recall_score(valid_y, y_pred)
        f1 = f1_score(valid_y, y_pred)
        
        print(f"Fold {n_fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        if show_feature_importance and hasattr(model, 'feature_importances_'):
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = model.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        gc.collect()
    
    print(f'Full AUC score: {roc_auc_score(train_df["TARGET"], oof_preds):.6f}')
    
    fig, axes = plt.subplots(num_folds, 2, figsize=(12, 5 * num_folds))
    
    for i in range(num_folds):
        ax1, ax2 = axes[i]
        
        ax1.plot(fpr_all[i], tpr_all[i], label=f'Fold {i + 1} (AUC = {roc_auc_all[i]:.2f})')
        ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - Fold {i + 1}')
        ax1.legend(loc='lower right')
        
        sns.heatmap(cm_all[i], annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        ax2.set_title(f'Confusion Matrix - Fold {i + 1}')
    
    plt.tight_layout()
    plt.show()
    
    if show_feature_importance and not feature_importance_df.empty:
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=feature_importance_df.groupby("feature").mean().reset_index().sort_values(by="importance", ascending=False))
        plt.title(f'Feature Importance - {model_type}')
        plt.show()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    return oof_preds, feature_importance_df

# Exemple d'appel de la fonction
# create_and_evaluate_model(df, 'target', model_type='XGBoost')

# def train_and_evaluate_model(df, target_column, model_type="LightGBM", random_state=42):
#     """
#     Entraîne et évalue un modèle de classification (Gradient Boosting, XGBoost, LightGBM, CatBoost).
    
#     Paramètres :
#         df : DataFrame contenant les features et la colonne cible
#         target_column : str, nom de la colonne cible
#         model_type : str, choix du modèle ("GradientBoosting", "XGBoost", "LightGBM", "CatBoost")
#         random_state : int, graine aléatoire pour reproductibilité

#     Retour :
#         Résultats des métriques et affichage des courbes.
#     """
#     # Séparation des données entre entraînement et test
#     train_df = df[df[target_column].notnull()]
#     test_df = df[df[target_column].isnull()]
    
#     # Séparation des features et de la cible
#     X = train_df.drop(columns=[target_column])
#     y = train_df[target_column]
    
#     # Imputation des valeurs manquantes (sans modifier le DataFrame original)
#     imputer = SimpleImputer(strategy='median')
#     X_imputed = imputer.fit_transform(X)
    
#     # Séparation des données en train et validation (80/20)
#     X_train, X_valid, y_train, y_valid = train_test_split(X_imputed, y, test_size=0.2, stratify=y, random_state=random_state)
    
#     # Choix du modèle
#     if model_type == "GradientBoosting":
#         model = GradientBoostingClassifier(random_state=random_state)
#     elif model_type == "XGBoost":
#         model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
#     elif model_type == "LightGBM":
#         model = lgb.LGBMClassifier(random_state=random_state)
#     else:
#         raise ValueError("Modèle non supporté. Choisir parmi 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'.")

#     # Entraînement du modèle
#     model.fit(X_train, y_train)

#     # Prédictions
#     y_pred = model.predict(X_valid)
#     y_proba = model.predict_proba(X_valid)[:, 1]

#     # Calcul des métriques
#     accuracy = accuracy_score(y_valid, y_pred)
#     precision = precision_score(y_valid, y_pred)
#     recall = recall_score(y_valid, y_pred)
#     f1 = f1_score(y_valid, y_pred)
#     auc_roc = roc_auc_score(y_valid, y_proba)

#     # Validation croisée avec 5 folds
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#     cv_scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="accuracy")
    
#     # Affichage des résultats
#     print(f"Modèle: {model_type}")
#     print(f"Exactitude: {accuracy:.4f}")
#     print(f"Précision: {precision:.4f}")
#     print(f"Rappel: {recall:.4f}")
#     print(f"F-mesure: {f1:.4f}")
#     print(f"AUC-ROC: {auc_roc:.4f}")
#     print(f"Validation croisée (accuracy): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#     # Matrice de confusion
#     conf_matrix = confusion_matrix(y_valid, y_pred)
    
#     # Affichage des graphiques (matrice de confusion + courbe ROC)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Matrice de confusion
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
#     axes[0].set_title("Matrice de Confusion")
#     axes[0].set_xlabel("Prédit")
#     axes[0].set_ylabel("Réel")

#     # Courbe ROC
#     RocCurveDisplay.from_predictions(y_valid, y_proba, ax=axes[1])
#     axes[1].set_title("Courbe ROC")

#     plt.tight_layout()
#     plt.show()

#     return model

# def train_and_evaluate_model(df, target_column, model_type="LightGBM", random_state=42):
#     """
#     Fonction pour entraîner et évaluer un modèle de classification sur un dataset.

#     Args:
#         df (pd.DataFrame): Jeu de données contenant les features et la cible.
#         target_column (str): Nom de la colonne cible.
#         model_type (str): Type de modèle ('GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost').
#         random_state (int): Graine aléatoire pour la reproductibilité.

#     Returns:
#         None (Affiche les métriques et visualisations).
#     """
#     # Supprimer les colonnes entièrement vides
#     df = df.dropna(axis=1, how='all')

#     # Supprimer les lignes où la cible est NaN
#     df = df.dropna(subset=[target_column])

#     # Séparation des features et de la cible
#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     # Vérifier qu'il ne reste pas que des NaN après le drop
#     if y.isna().sum() > 0:
#         raise ValueError("Il reste des NaN dans la cible après suppression. Vérifiez vos données.")

#     # Vérifier qu'il y a au moins deux classes dans y
#     if len(np.unique(y)) < 2:
#         raise ValueError("La colonne cible doit contenir au moins deux classes distinctes.")

#     # Séparation en jeu d'entraînement et de validation (80%-20%)
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

#     # Convertir en DataFrame pour conserver les noms des features (important pour LightGBM)
#     X_train = pd.DataFrame(X_train, columns=X.columns)
#     X_valid = pd.DataFrame(X_valid, columns=X.columns)

#     # Imputation des valeurs manquantes avec la médiane
#     imputer = SimpleImputer(strategy="median")
#     X_train_imputed = imputer.fit_transform(X_train)
#     X_valid_imputed = imputer.transform(X_valid)
#     X_imputed = imputer.fit_transform(X)  # Utilisé pour la validation croisée

#     # Définition du modèle en fonction du type spécifié
#     if model_type == "GradientBoosting":
#         model = GradientBoostingClassifier(random_state=random_state)
#     elif model_type == "XGBoost":
#         model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)
#     elif model_type == "LightGBM":
#         scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Gestion du déséquilibre
#         model = LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=random_state)
#     elif model_type == "CatBoost":
#         model = CatBoostClassifier(verbose=0, random_state=random_state)
#     else:
#         raise ValueError("Modèle non reconnu. Choisissez parmi: 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost'.")

#     # Entraînement du modèle
#     model.fit(X_train_imputed, y_train)

#     # Prédictions
#     y_pred = model.predict(X_valid_imputed)
#     y_proba = model.predict_proba(X_valid_imputed)[:, 1]

#     # Calcul des métriques
#     accuracy = accuracy_score(y_valid, y_pred)
#     precision = precision_score(y_valid, y_pred)
#     recall = recall_score(y_valid, y_pred)
#     f1 = f1_score(y_valid, y_pred)
#     auc_roc = roc_auc_score(y_valid, y_proba)

#     # Validation croisée (5 folds stratifiés)
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#     cv_scores = cross_val_score(model, X_imputed, y, cv=cv, scoring="accuracy")

#     # Affichage des résultats
#     print(f"\nModèle: {model_type}")
#     print(f"Exactitude (Accuracy): {accuracy:.4f}")
#     print(f"Précision: {precision:.4f}")
#     print(f"Rappel: {recall:.4f}")
#     print(f"F-mesure: {f1:.4f}")
#     print(f"AUC-ROC: {auc_roc:.4f}")
#     print(f"Validation croisée (accuracy): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

#     # Matrice de confusion
#     conf_matrix = confusion_matrix(y_valid, y_pred)

#     # Affichage graphique
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Matrice de confusion
#     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Classe 0", "Classe 1"], yticklabels=["Classe 0", "Classe 1"], ax=axes[0])
#     axes[0].set_title("Matrice de confusion")
#     axes[0].set_xlabel("Prédiction")
#     axes[0].set_ylabel("Réel")

#     # Courbe ROC
#     RocCurveDisplay.from_estimator(model, X_valid_imputed, y_valid, ax=axes[1])
#     axes[1].set_title("Courbe ROC")

#     plt.tight_layout()
#     plt.show()

def train_and_evaluate_model(X, y, model=None, test_size=0.2, cv=5, scale_pos_weight=None):
    """
    Entraîne et évalue un modèle de classification avec gestion des valeurs manquantes et affichage des métriques.

    Paramètres :
    - X : DataFrame des features
    - y : Série cible
    - model : Modèle à entraîner (LightGBM par défaut)
    - test_size : Fraction du dataset pour le test
    - cv : Nombre de folds pour la validation croisée
    - scale_pos_weight : Poids à attribuer à la classe minoritaire pour gérer le déséquilibre des classes

    Retourne :
    - Le modèle entraîné et les scores d’évaluation
    """
    # 1️⃣ Suppression des colonnes vides
    X = X.dropna(axis=1, how='all')

    # 2️⃣ Séparation des données
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # 3️⃣ Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    # 4️⃣ Conversion en DataFrame pour éviter les warnings
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_valid_imputed = pd.DataFrame(X_valid_imputed, columns=X_valid.columns)

    # 5️⃣ Initialisation du modèle si non fourni
    if model is None:
        model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)

    # 6️⃣ Entraînement du modèle
    model.fit(X_train_imputed, y_train)

    # 7️⃣ Prédictions
    y_pred = model.predict(X_valid_imputed)
    y_pred_proba = model.predict_proba(X_valid_imputed)[:, 1]

    # 8️⃣ Évaluation des performances
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    auc_roc = roc_auc_score(y_valid, y_pred_proba)

    # Validation croisée
    cv_scores = cross_val_score(model, X_train_imputed, y_train, cv=cv, scoring="accuracy")

    # 🔹 Affichage des métriques
    print(f"🔹 Modèle: {model.__class__.__name__}")
    print(f"✅ Exactitude (Accuracy): {accuracy:.4f}")
    print(f"✅ Précision: {precision:.4f}")
    print(f"✅ Rappel: {recall:.4f}")
    print(f"✅ F-mesure: {f1:.4f}")
    print(f"✅ AUC-ROC: {auc_roc:.4f}")
    print(f"✅ Validation croisée (accuracy): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 9️⃣ Affichage de la matrice de confusion et de la courbe ROC
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Matrice de confusion
    cm = confusion_matrix(y_valid, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0])
    ax[0].set_title("Matrice de confusion")
    ax[0].set_xlabel("Prédit")
    ax[0].set_ylabel("Réel")

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_valid, y_pred_proba)
    ax[1].plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_title("Courbe ROC")
    ax[1].set_xlabel("Taux de faux positifs")
    ax[1].set_ylabel("Taux de vrais positifs")
    ax[1].legend()

    plt.show()

    return model