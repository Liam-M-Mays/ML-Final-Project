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
git clone https://github.com/Liam-M-Mays/ML-Final-Project.git
cd "ML Final Project"
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

Run the notebook to regenerate `california_housing_report.pdf` (there is no separate `generate_report.py` script):

```bash
jupyter nbconvert --to notebook --execute --output=california_housing_prediction.ipynb california_housing_prediction.ipynb
jupyter nbconvert --to pdf california_housing_prediction.ipynb
```

## Project Structure

```
ML Final Project/
├── california_housing_prediction.ipynb  # Main analysis notebook
├── california_housing_report.pdf        # Academic project report
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── figures/                             # Generated visualizations
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

## Source Code

Public repository (required for grading): https://github.com/Liam-M-Mays/ML-Final-Project.git
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

## References (MLA)

- Breiman, Leo. "Random Forests." *Machine Learning*, vol. 45, no. 1, 2001, pp. 5-32.
- Friedman, Jerome H. "Greedy Function Approximation: A Gradient Boosting Machine." *The Annals of Statistics*, vol. 29, no. 5, 2001, pp. 1189-1232.
- Hastie, Trevor, et al. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed., Springer, 2009.
- Pace, R. Kelley, and Ronald Barry. "Sparse Spatial Autoregressions." *Statistics & Probability Letters*, vol. 33, no. 3, 1997, pp. 291-297.
- Pedregosa, Fabian, et al. "Scikit-Learn: Machine Learning in Python." *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825-2830.
- "California Housing Dataset." *Scikit-learn Documentation*, scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset. Accessed December 2024.

## Acknowledgements and AI Use

The author acknowledges the use of ChatGPT in the preparation or completion of this assignment. ChatGPT was used for brainstorming report wording, proofreading for grammar and clarity. Claude AI (Anthropic) assisted with code organization and documentation. All analysis, modeling decisions, and interpretation of results were performed by the author.

- Dataset: California Housing from scikit-learn (derived from 1990 U.S. Census)

## Author

Liam Mays
ITCS 3156
UNC Charlotte
December 2024
