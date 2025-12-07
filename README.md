# California Housing Price Prediction

A comprehensive machine learning project that compares multiple regression algorithms for predicting housing prices in California using census data.

## Project Description

This project applies machine learning techniques to predict median housing prices in California districts based on demographic and geographic features from the 1990 U.S. Census. The analysis includes exploratory data analysis, implementation of three regression algorithms, hyperparameter tuning, and comprehensive model evaluation.

### Key Features
- Complete exploratory data analysis with 10 visualizations
- Three regression algorithms with detailed rationale
- Hyperparameter optimization using GridSearchCV with 5-fold cross-validation
- Comprehensive model evaluation and comparison
- Feature importance analysis
- Professional PDF report following academic standards

## Dataset Information

**Source**: StatLib repository (1990 U.S. Census), available via scikit-learn

**Dataset Characteristics**:
- **Samples**: 20,640 California census block groups
- **Features**: 8 numeric attributes
- **Target**: Median house value (in $100,000s)
- **Missing Values**: None

**Features**:
| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

## Results Summary

### Model Performance (Test Set)

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.5758 | 0.7456 | 0.5332 |
| Random Forest | 0.8061 | 0.5040 | 0.3260 |
| **Gradient Boosting** | **0.8422** | **0.4548** | **0.2964** |

### Key Findings

1. **Best Model**: Gradient Boosting Regressor achieved the highest R² score of 0.8422
2. **Ensemble Superiority**: Ensemble methods outperform Linear Regression by ~27% in R²
3. **Most Important Feature**: Median Income (MedInc) is the strongest predictor across all models
4. **Geographic Impact**: Latitude and Longitude significantly influence house values (coastal premium)

## Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd MLproject
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Jupyter Notebook Interface

```bash
jupyter notebook
# Open california_housing_prediction.ipynb
# Run all cells: Cell -> Run All
```

### Option 2: Command Line Execution

```bash
jupyter nbconvert --to notebook --execute --inplace california_housing_prediction.ipynb
```

### Option 3: Generate PDF Report Only

```bash
python generate_report.py
```

## Project Structure

```
MLproject/
├── california_housing_prediction.ipynb  # Main analysis notebook
├── california_housing_report.pdf        # Academic project report
├── generate_report.py                   # PDF report generator
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── figures/                            # Generated visualizations
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_scatter_plots.png
│   ├── 05_geographic_distribution.png
│   ├── 06_boxplots.png
│   ├── 07_model_comparison.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_residual_analysis.png
│   └── 10_feature_importance_comparison.png
└── results/
    └── model_results.csv
```

## Methods

### 1. Linear Regression (Baseline)
- Simple parametric model assuming linear relationships
- Provides interpretable coefficients
- Fast training and prediction

### 2. Random Forest Regressor
- Ensemble of decision trees using bagging
- Captures non-linear relationships
- Provides feature importance rankings
- Hyperparameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf

### 3. Gradient Boosting Regressor
- Sequential ensemble using boosting
- Achieves state-of-the-art performance on tabular data
- Hyperparameters tuned: n_estimators, learning_rate, max_depth, subsample

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0
- fpdf2 >= 2.7.0

## References

1. Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
2. Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine." The Annals of Statistics, 29(5), 1189-1232.
3. Hastie, T., et al. (2009). The Elements of Statistical Learning. Springer.
4. Pace, R. K., & Barry, R. (1997). "Sparse Spatial Autoregressions." Statistics & Probability Letters, 33(3), 291-297.
5. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." JMLR, 12, 2825-2830.

## Acknowledgements

- Dataset: California Housing from scikit-learn (derived from 1990 U.S. Census)
- AI Assistance: Claude AI (Anthropic) was used for code organization and documentation. All code and content was generated by the author

## Author

Liam Mays
ITCS 3156
UNC Charlotte
December 2024

