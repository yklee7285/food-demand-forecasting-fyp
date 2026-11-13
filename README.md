# Food Demand Forecasting System for a Campus Canteen

Final-year project for forecasting daily meal demand at a university student canteen, as a step towards reducing food waste.

## Notebooks

| Notebook | What it does | Open |
|---|---|---|
| `synthetic_canteen_data_generation.ipynb` | Generates and validates a synthetic daily sales dataset for the canteen (2025–2026, Monday–Saturday) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/synthetic_canteen_data_generation.ipynb) |
| `canteen_methodology.ipynb` | Forecasts daily meals sold with Random Forest, XGBoost, SVR, LSTM and two stacking ensembles, and compares their accuracy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/canteen_methodology.ipynb) |

## Data

| File | Description |
|---|---|
| `datasets/canteen_sales_2025_2026.xlsx` | Synthetic canteen dataset produced by the generation notebook (626 operating days) |
| `datasets/meal_delivery_dataset.csv` | Additional dataset for generalisability testing (source to be added) |

The canteen dataset is **synthetic**: it is generated from stated assumptions, not recorded sales.

## Running

**Google Colab:** click an "Open in Colab" badge above. The forecasting notebook clones this repository automatically to load the dataset.

**Locally:** clone the repository, install the requirements and open the notebooks with Jupyter:

```bash
pip install pandas numpy scipy scikit-learn xgboost tensorflow matplotlib seaborn openpyxl
```

The synthetic dataset is already included in `datasets/`, and running `synthetic_canteen_data_generation.ipynb` will regenerate the exact same dataset through the same random seed.
