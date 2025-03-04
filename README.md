# Principal Component Analysis (PCA) API

This Django application provides a REST API for performing Principal Component Analysis on numerical data.

## Features

- From-scratch PCA implementation
- Support for multiple analysis types:
  - Normalized PCA
  - Homogeneous PCA
  - Heterogeneous PCA
- CSV file upload capability

## Installation

1. Clone repository
```bash
git clone https://github.com/Ratybox/madabina.git
cd pca_tp_and
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start Django server
```bash
python manage.py runserver
```

## Project Structure

- `backend/`: Core Django application
  - `utils.py`: PCA algorithms implementation
  - `views.py`: API endpoint handlers
  - `urls.py`: Route configuration
- `pca_tp_and/`: Django project configuration

## API Endpoints

### CSV File Upload
- **URL**: `/api/upload-csv-data/`
- **Method**: `POST`
- **Parameters**:
  - `file`: CSV file to analyze
- **Response**: Upload confirmation and basic data info

### General PCA Analysis
- **URL**: `/api/perform-pca/`
- **Method**: `POST`
- **Parameters**:
  - `pca_type`: Analysis type ('normalized', 'homogeneous', 'heterogeneous')
- **Response**: Full PCA results

### Homogeneous PCA
- **URL**: `/api/perform-pca-homogeneous/`
- **Method**: `POST`
- **Response**: Homogeneous PCA results

### Heterogeneous PCA 
- **URL**: `/api/perform-pca-heterogeneous/`
- **Method**: `POST`
- **Response**: Heterogeneous PCA results

### Simple Data Check
- **URL**: `/api/get-data/`
- **Method**: `GET`
- **Response**: Basic API status message

## Response Format

Typical response contains:
```json
{
  "pca_type": "normalized",
  "n_individuals": 150,
  "n_variables": 4,
  "variables": ["sepal_length", "sepal_width", ...],
  "eigenvalues": [2.918, 0.914, ...],
  "inertia": {
    "total": 4.573,
    "explained_percent_by_axis": [63.8, 23.9, ...]
  },
  "principal_components": [[-2.684, 0.319, ...], ...],
  "correlations": [[0.890, -0.034, ...], ...],
  "contributions": {
    "variables": [[45.2, 1.8, ...], ...],
    "individuals": [[0.52, 0.03, ...], ...]
  }
}
```