# Principal Component Analysis (PCA) API

This Django application provides a REST API for performing Principal Component Analysis on numerical data.

## Features

- From-scratch PCA implementation (no external libraries)
- Support for multiple analysis types:
  - ✅ Normalized PCA
  - ⚖️ Homogeneous PCA
  - ⚡ Heterogeneous PCA
- CSV file upload capability

## Installation

1. **Clone repository**
```bash
git clone https://github.com/Ratybox/madabina.git
cd pca_tp_and
```

2. **Set up virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Django server**
```bash
python manage.py runserver
```

## Response Format
```json
{
  "pca_type": "normalized",
  "n_individuals": 150,
  "n_variables": 4,
  "variables": ["sepal_length", "sepal_width", ...],
  "eigenvalues": [2.918, 0.914, ...],
  "inertia": {
    "total": 4.573,
    "explained_percent_by_axis": [63.8, 23.9, ...],
    "cumulative_percent": [63.8, 87.7, ...]
  },
  "principal_components": [[-2.684, 0.319, ...], ...],
  "correlations": [[0.890, -0.034, ...], ...],
  "contributions": {
    "variables": [[45.2, 1.8, ...], ...],
    "individuals": [[0.52, 0.03, ...], ...]
  }
}
```

## Project Structure
```
pca_tp_and/
├── backend/
│   ├── utils.py       # PCA algorithm implementation
│   ├── views.py       # API endpoints
│   └── urls.py        # Route configuration
└── requirements.txt   # Dependencies