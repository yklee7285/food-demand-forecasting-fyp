# Food Waste Forecasting System for Campus Canteens

Final-year project that forecasts food demand and food waste so that canteens can prepare closer to what is actually needed and waste less food.

The project replicates a published stacking-ensemble method ([Seyam et al., 2025](https://doi.org/10.1016/j.clscn.2025.100225)) and then improves it. The improved version removes data leakage, trains the stacking ensembles faster, tests the method on three public datasets and explains the forecasts with SHAP and LIME.

## Notebooks

The notebooks are in `notebooks/`, one folder per dataset. Each dataset has two notebooks: a replication of the original method and an improved version.

| Notebook | Data | Open |
|---|---|---|
| `synthetic_canteen/synthetic_canteen_data_generation.ipynb` | Creates the synthetic canteen dataset | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/synthetic_canteen/synthetic_canteen_data_generation.ipynb) |
| `synthetic_canteen/canteen_methodology.ipynb` | Synthetic canteen: daily meals sold (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/synthetic_canteen/canteen_methodology.ipynb) |
| `synthetic_canteen/canteen_methodology_improved.ipynb` | Synthetic canteen: daily meals sold (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/synthetic_canteen/canteen_methodology_improved.ipynb) |
| `kaggle_canteen/kaggle_canteen_methodology.ipynb` | University canteen: daily food waste (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/kaggle_canteen/kaggle_canteen_methodology.ipynb) |
| `kaggle_canteen/kaggle_canteen_methodology_improved.ipynb` | University canteen: daily food waste (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/kaggle_canteen/kaggle_canteen_methodology_improved.ipynb) |
| `meal_delivery/meal_delivery_methodology.ipynb` | Meal delivery company: weekly orders (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/meal_delivery/meal_delivery_methodology.ipynb) |
| `meal_delivery/meal_delivery_methodology_improved.ipynb` | Meal delivery company: weekly orders (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/meal_delivery/meal_delivery_methodology_improved.ipynb) |
| `restaurant/restaurant_methodology.ipynb` | Restaurant events: food waste (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/restaurant/restaurant_methodology.ipynb) |
| `restaurant/restaurant_methodology_improved.ipynb` | Restaurant events: food waste (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/notebooks/restaurant/restaurant_methodology_improved.ipynb) |

Models used: Random Forest, XGBoost, SVR, LSTM and two stacking ensembles.

## Dashboard

`dashboard/` contains a Streamlit dashboard for canteen staff. Each day it shows how many meals to prepare and why; at the end of the day staff record the meals sold, and the next forecast uses them. It also tracks leftovers, flags unusual days, prints a kitchen prep sheet and has a what-if tool and a performance page. It uses the stacking ensemble (Random Forest, XGBoost and SVR) trained in `notebooks/synthetic_canteen/canteen_methodology_improved.ipynb`.

On Windows, double-click `run_dashboard.bat` (needs Python 3.13). Or run:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

More options, including an online demo, are in `dashboard/README.md`.

## Data

All datasets are in `datasets/`.

- `canteen_sales_2025_2026.xlsx`: synthetic data created for this project, not real sales.
- `canteen_dataset.csv`: [Food Waste Dataset from a University Canteen](https://www.kaggle.com/datasets/rahuljangir78/food-waste-dataset-from-a-university-canteen) by rahuljangir78 on Kaggle (Apache 2.0).
- `meal_delivery_dataset.csv`: [Food Demand Forecasting](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting) by Edwin U Kannanaikkal on Kaggle, originally from Analytics Vidhya (Open Database License).
- `restaurant_dataset.csv`: [Food wastage data in restaurant](https://www.kaggle.com/datasets/trevinhannibal/food-wastage-data-in-restaurant) by Trevin Hannibal on Kaggle.

The Kaggle datasets belong to their original authors and are included only so the notebooks can be run.

## Running

Click an "Open in Colab" button above. The notebooks load the data from this repository automatically.

To run locally:

```bash
git clone https://github.com/yklee7285/food-demand-forecasting-fyp.git
cd food-demand-forecasting-fyp
pip install -r requirements.txt
```
