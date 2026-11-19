# Food Waste Prediction System for Campus Canteens

Final-year project that forecasts food demand and food waste so that canteens can prepare closer to what is actually needed and waste less food.

The project replicates a published stacking-ensemble method ([Seyam et al., 2025](https://doi.org/10.1016/j.clscn.2025.100225)) and then improves it. The improved version removes data leakage, trains the stacking ensembles faster, tests the method on three public datasets and explains the forecasts with SHAP and LIME.

## Notebooks

Each dataset has two notebooks: a replication of the original method and an improved version.

| Notebook | Data | Open |
|---|---|---|
| `synthetic_canteen_data_generation.ipynb` | Creates the synthetic canteen dataset | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/synthetic_canteen_data_generation.ipynb) |
| `canteen_methodology.ipynb` | Synthetic canteen: daily meals sold (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/canteen_methodology.ipynb) |
| `canteen_methodology_improved.ipynb` | Synthetic canteen: daily meals sold (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/canteen_methodology_improved.ipynb) |
| `kaggle_canteen_methodology.ipynb` | University canteen: daily food waste (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/kaggle_canteen_methodology.ipynb) |
| `kaggle_canteen_methodology_improved.ipynb` | University canteen: daily food waste (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/kaggle_canteen_methodology_improved.ipynb) |
| `meal_delivery_methodology.ipynb` | Meal delivery company: weekly orders (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/meal_delivery_methodology.ipynb) |
| `meal_delivery_methodology_improved.ipynb` | Meal delivery company: weekly orders (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/meal_delivery_methodology_improved.ipynb) |
| `restaurant_methodology.ipynb` | Restaurant events: food waste (replication) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/restaurant_methodology.ipynb) |
| `restaurant_methodology_improved.ipynb` | Restaurant events: food waste (improved) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yklee7285/food-demand-forecasting-fyp/blob/main/restaurant_methodology_improved.ipynb) |

Models used: Random Forest, XGBoost, SVR, LSTM and two stacking ensembles.

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
