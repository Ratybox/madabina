# API d'Analyse en Composantes Principales (PCA)

Cette application Django fournit une API REST permettant de réaliser des analyses en composantes principales (PCA) sur des données numériques.

## Fonctionnalités

- Implémentation "from scratch" des méthodes PCA (sans utiliser sklearn)
- Support de plusieurs types d'analyses :
  - PCA normée
  - PCA non normée homogène
  - PCA non normée hétérogène
- Upload de fichiers CSV
- Calcul complet des composantes principales, valeurs propres, vecteurs propres, etc.
- Calcul des métriques de qualité, inertie, et contributions

## Installation

1. Cloner le dépôt
```bash
git clone <URL_DU_REPO>
cd pca_tp_and
```

2. Installer les dépendances
```bash
pip install -r requirements.txt
```

3. Démarrer le serveur Django
```bash
python manage.py runserver
```

## Structure du projet

- `backend/` : Application Django principale
  - `utils.py` : Implémentation des algorithmes PCA
  - `views.py` : Endpoints API REST
  - `urls.py` : Configuration des routes
- `pca_tp_and/` : Configuration du projet Django

## API Endpoints

### Upload de fichier CSV
- **URL** : `/api/upload-csv/`
- **Méthode** : `POST`
- **Paramètres** :
  - `file` : Fichier CSV à analyser
  - `separator` (optionnel) : Séparateur CSV (défaut: ',')
- **Réponse** : Informations sur les données chargées

### Analyse PCA sur les données chargées
- **URL** : `/api/perform-pca/`
- **Méthode** : `POST`
- **Paramètres** :
  - `pca_type` : Type d'analyse ('normalized', 'homogeneous', 'heterogeneous')
- **Réponse** : Résultats complets de l'analyse PCA

### Analyse PCA sur des données brutes
- **URL** : `/api/analyze-data/`
- **Méthode** : `POST`
- **Paramètres** :
  - `data` : Tableau 2D de données numériques
  - `pca_type` (optionnel) : Type d'analyse (défaut: 'normalized')
- **Réponse** : Résultats complets de l'analyse PCA

### Test avec données d'exemple
- **URL** : `/api/test-example/`
- **Méthode** : `GET`
- **Réponse** : Résultats des trois types d'analyses PCA sur les données d'exemple

## Format de réponse

La réponse contient un dictionnaire complet avec :
- `pca_type` : Type d'analyse effectuée
- `n_individuals`, `n_variables` : Dimensions des données
- `variables`, `individuals` : Noms des variables et individus
- `data_original`, `data_centered` : Données originales et centrées
- `variance_matrix`, `metric` : Matrices de variance et métrique
- `eigenvalues`, `eigenvectors` : Valeurs et vecteurs propres
- `inertia` : Informations sur l'inertie totale et par axe
- `n_significant_components` : Nombre de composantes significatives
- `principal_components` : Composantes principales
- `correlations` : Corrélations entre variables et axes
- `contributions` : Contributions des variables et individus
- `cos2` : Qualité de représentation des individus

## Exemple d'utilisation

### Python (avec requests)
```python
import requests
import json

# Upload d'un fichier CSV
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload-csv/', 
                           files={'file': f})
print(response.json())

# Analyse PCA
response = requests.post('http://localhost:8000/api/perform-pca/',
                       json={'pca_type': 'normalized'})
results = response.json()
print(results) 