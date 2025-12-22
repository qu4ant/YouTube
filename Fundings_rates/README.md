# Analyse des Funding Rates et Basis Crypto

Analyse des funding rates et du basis (spread futures/spot) pour le trading de cryptomonnaies. Permet de detecter les zones de panique et d'euphorie sur le marche.

## Fonctionnalites

- Calcul du basis (ecart entre prix futures et spot)
- Analyse des funding rates
- Detection de sentiment via Z-score (panique/euphorie)
- Visualisations interactives avec Bokeh (theme sombre)
- Support multi-actifs : BTC, ETH, AAVE

## Structure du projet

```
.
├── basis_funding_analysis.ipynb   # Notebook principal d'analyse
├── crypto.db                       # Base de donnees DuckDB
├── pyproject.toml                  # Configuration projet (uv/pip)
├── requirements.txt                # Dependances pip/conda
├── environment.yml                 # Environnement conda
└── README.md
```

## Installation

### Option 1 : UV (recommande)

```bash
# Installer uv si necessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Creer l'environnement et installer les dependances
uv sync

# Lancer Jupyter
uv run jupyter notebook
```

### Option 2 : Conda

```bash
# Creer l'environnement
conda env create -f environment.yml

# Activer l'environnement
conda activate funding-rates

# Lancer Jupyter
jupyter notebook
```

### Option 3 : pip

```bash
# Creer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer les dependances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook
```

## Utilisation

1. Lancer Jupyter Notebook
2. Ouvrir `basis_funding_analysis.ipynb`
3. Configurer les parametres dans la premiere cellule :
   - `symbol` : BTCUSDT, ETHUSDT, AAVEUSDT
   - `start_date` / `end_date` : periode d'analyse
   - `sma_period` : periode de la moyenne mobile
   - `zscore_window` : fenetre pour le calcul du Z-score
   - `zscore_threshold` : seuil de detection (defaut: 2.0)
4. Executer toutes les cellules

## Indicateurs

- **Basis** : `(prix_futures - prix_spot) / prix_spot * 100`
- **Z-score** : Mesure l'ecart par rapport a la moyenne sur une fenetre glissante
  - Z-score < -2 : Zone de panique (rouge)
  - Z-score > +2 : Zone d'euphorie (vert)

## Dependances

- duckdb >= 0.8.0
- pandas >= 1.3.0
- numpy >= 1.20.0
- bokeh >= 3.0.0
- jupyter >= 1.0.0

## Licence

MIT
