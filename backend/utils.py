import numpy as np
import pandas as pd
import json

def calculate_inertia(eigenvalues):
    return np.sum(eigenvalues)

def calculate_quality(eigenvalues, n_components=None):
    total_inertia = calculate_inertia(eigenvalues)
    
    if n_components is None:
        return np.ones(len(eigenvalues)) * 100
    
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    return cumulative_quality[:n_components]

def pca(data, pca_type='normalized'):
  
    # Convertir en numpy array si nécessaire
    if isinstance(data, pd.DataFrame):
        variables = data.columns.tolist()
        individuals = data.index.tolist()
        X = data.values
    else:
        X = np.array(data)
        variables = [f'V{i+1}' for i in range(X.shape[1])]
        individuals = [f'P{i+1}' for i in range(X.shape[0])]
    
    # Centrer les données
    X_centered = X - np.mean(X, axis=0)
    
    # Calculer la matrice de variance/covariance
    n = X.shape[0]
    variance_matrix = (1 / n) * (X_centered.T @ X_centered)
    
    # Définir la matrice métrique selon le type de PCA
    if pca_type == 'normalized':
        # PCA normée: Utilise la matrice diagonale des inverses des variances
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    elif pca_type == 'homogeneous':
        # PCA non normée homogène: Utilise la matrice identité
        metric = np.eye(X.shape[1])
    elif pca_type == 'heterogeneous':
        # PCA non normée hétérogène: Utilise une autre pondération
        std = np.std(X_centered, axis=0, ddof=1)
        metric = np.diag(1 / std**2)
    else:
        raise ValueError("Type de PCA non reconnu. Utilisez 'normalized', 'homogeneous' ou 'heterogeneous'")
    
    # Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(metric @ variance_matrix)
    
    # Tri par ordre décroissant
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calcul de la qualité et du nombre d'axes à retenir (critère 80%)
    total_inertia = calculate_inertia(eigenvalues)
    cumulative_quality = np.cumsum(eigenvalues) / total_inertia * 100
    n_components = np.argmax(cumulative_quality >= 80) + 1 if np.any(cumulative_quality >= 80) else len(eigenvalues)
    
    # Calculer les composantes principales
    principal_components = X_centered @ eigenvectors
    
    # Corrélations entre variables et composantes
    correlations = np.zeros((X.shape[1], len(eigenvalues)))
    for i in range(X.shape[1]):
        for j in range(len(eigenvalues)):
            sigma_x = np.std(X_centered[:, i])
            correlations[i, j] = np.sum(X_centered[:, i] * principal_components[:, j]) / (n * sigma_x * np.sqrt(eigenvalues[j]))
    
    # Contributions des variables aux axes
    contributions_var = np.zeros((X.shape[1], len(eigenvalues)))
    for i in range(X.shape[1]):
        for j in range(len(eigenvalues)):
            contributions_var[i, j] = (eigenvectors[i, j] ** 2) * eigenvalues[j] / total_inertia * 100
    
    # Contributions des individus aux axes
    contributions_ind = np.zeros((X.shape[0], len(eigenvalues)))
    for i in range(X.shape[0]):
        for j in range(len(eigenvalues)):
            contributions_ind[i, j] = (principal_components[i, j] ** 2) / (eigenvalues[j])
    
    # Qualité de représentation des individus
    cos2_ind = np.zeros((X.shape[0], len(eigenvalues)))
    for i in range(X.shape[0]):
        for j in range(len(eigenvalues)):
            cos2_ind[i, j] = (principal_components[i, j] ** 2) / np.sum(X_centered[i] ** 2)
    
    # Préparer les résultats
    results = {
        'pca_type': pca_type,
        'n_individuals': X.shape[0],
        'n_variables': X.shape[1],
        'variables': variables,
        'individuals': individuals,
        'data_original': X.tolist(),
        'data_centered': X_centered.tolist(),
        'variance_matrix': variance_matrix.tolist(),
        'metric': metric.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'inertia': {
            'total': float(total_inertia),
            'explained_by_axis': eigenvalues.tolist(),
            'explained_percent_by_axis': (eigenvalues / total_inertia * 100).tolist(),
            'cumulative_percent': cumulative_quality.tolist()
        },
        'n_significant_components': int(n_components),
        'principal_components': principal_components.tolist(),
        'correlations': correlations.tolist(),
        'contributions': {
            'variables': contributions_var.tolist(),
            'individuals': contributions_ind.tolist()
        },
        'cos2': {
            'individuals': cos2_ind.tolist()
        }
    }
    
    return results


def prepare_response(results):
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results[key] = value.tolist()
        elif isinstance(value, dict):
            results[key] = prepare_response(value)
    
    return results
