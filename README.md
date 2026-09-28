# Student Performance Regression

A machine-learning project that predicts students’ final mathematics grade (`G3`) from demographic, family, school, lifestyle, and academic information. The project uses exploratory data analysis, preprocessing, anomaly handling, and regression-model comparison to show how access to prior-period grades changes predictive performance.

> **Educational use only.** This project analyses a small, historical student survey dataset. Its predictions should not be used to make high-stakes decisions about individual students.

## Overview

The notebook investigates a supervised regression problem: estimating a secondary-school student’s final mathematics grade on a scale from 0 to 20.

Two modelling scenarios are compared:

- **Scenario A — with prior grades:** Uses first-period (`G1`) and second-period (`G2`) grades as predictors.
- **Scenario B — without prior grades:** Excludes `G1` and `G2` to assess how well final performance can be estimated from background, school, family, and behavioural variables alone.

This comparison is important because prior grades are highly informative but may not be available early enough for some intervention use cases.

## Dataset

The dataset contains survey responses from **395 secondary-school students** enrolled in mathematics. It includes **33 original features** covering:

| Category | Example variables |
|---|---|
| Demographic information | `school`, `sex`, `age`, `address`, `famsize`, `Pstatus` |
| Parent and family background | `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian`, `famrel` |
| Education and support | `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `higher` |
| Lifestyle and wellbeing | `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic` |
| Attendance and academic history | `absences`, `G1`, `G2` |
| Target | `G3` — final mathematics grade from 0 to 20 |

### Target summary

| Statistic | Final grade (`G3`) |
|---|---:|
| Number of students | 395 |
| Mean | 10.42 |
| Median | 11.00 |
| Minimum | 0 |
| Maximum | 20 |

The notebook identifies records with a final grade of 0 and examines the relationship between final grades, prior grades, attendance, failures, and other student characteristics.

## Project workflow

The main notebook, `Student-Performance-Regression.ipynb`, implements the following workflow:

1. **Load and inspect data**
   - Import the student-performance dataset.
   - Review sample records, data types, descriptive statistics, and missing values.

2. **Exploratory data analysis**
   - Explore numeric distributions and categorical features.
   - Examine grade distributions and the presence of zero final grades.
   - Investigate relationships between predictors and final performance.

3. **Data preparation**
   - Separate predictors from the final-grade target (`G3`).
   - Encode categorical variables using one-hot encoding.
   - Apply feature scaling where required.
   - Use DBSCAN as part of the exploratory/anomaly-detection workflow.

4. **Model training**
   - Split data into training and evaluation sets.
   - Train several regression models:
     - Linear Regression
     - Decision Tree Regressor
     - Random Forest Regressor
     - Gradient Boosting Regressor
   - Use cross-validation utilities to support model assessment.

5. **Model evaluation**
   - Compare models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and coefficient of determination (
     \(R^2\)).
   - Compare predictive accuracy with and without `G1` and `G2`.

## Results

The notebook compares four models under both scenarios.

| Scenario | Model | MAE | RMSE | \(R^2\) |
|---|---|---:|---:|---:|
| A — with `G1`, `G2` | Linear Regression | **0.6533** | **0.8484** | **0.9307** |
| A — with `G1`, `G2` | Decision Tree | 0.8351 | 1.1817 | 0.8660 |
| A — with `G1`, `G2` | Random Forest | 0.7056 | 0.9095 | 0.9208 |
| A — with `G1`, `G2` | Gradient Boosting | 0.7197 | 0.9655 | 0.9105 |
| B — without `G1`, `G2` | Linear Regression | **2.3845** | **2.9859** | **0.1533** |
| B — without `G1`, `G2` | Decision Tree | 3.4667 | 4.4089 | -0.8617 |
| B — without `G1`, `G2` | Random Forest | 2.4916 | 3.1313 | 0.0713 |
| B — without `G1`, `G2` | Gradient Boosting | 2.6589 | 3.4367 | -0.1204 |

### Key findings

- **Linear Regression** achieved the best reported performance in both scenarios.
- With `G1` and `G2` included, Linear Regression reached an \(R^2\) of **0.9307** and an RMSE of **0.8484**, indicating that prior assessment grades explain much of the variation in final grades.
- Without `G1` and `G2`, the best \(R^2\) fell to **0.1533** and the RMSE increased to **2.9859**.
- The large performance gap shows that prior academic performance is substantially more predictive of final grade than the remaining survey features alone.

## Repository structure

```text
.
├── Student-Performance-Regression.ipynb   # EDA, preprocessing, model training, and evaluation
└── README.md                               # Project documentation
```

## Requirements

Install the libraries used in the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Getting started

1. Clone or download the repository.
2. Create and activate a Python virtual environment.
3. Install the project dependencies.
4. Open the Jupyter notebook.
5. Ensure the source dataset is available at the path configured in the notebook.
6. Run the cells in order to reproduce the analysis and model results.

Example environment setup:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

## Metrics

The regression models are evaluated with the following metrics:

- **MAE:** Mean absolute difference between predicted and actual final grades. Lower is better.
- **RMSE:** Square-root of the mean squared prediction error. It penalises larger errors more strongly than MAE. Lower is better.
- **\(R^2\):** Proportion of target variation explained by the model. Values closer to 1 indicate better fit; negative values indicate performance worse than predicting the mean.

For example, Scenario A’s best model has an MAE of about 0.65 grade points, while the best model without prior grades has an MAE of about 2.38 grade points.

## Limitations

- The dataset is small, with 395 observations, so results may vary across train/test splits.
- The data represent mathematics students from one historical survey context and may not generalise to other schools, countries, cohorts, or subjects.
- Prior grades (`G1` and `G2`) are strong predictors but may introduce information-timing concerns if the intended use is early intervention.
- Correlation and predictive association do not establish that a feature causes a change in student achievement.
- Student data are sensitive. Any real-world application should address privacy, fairness, transparency, and the risk of harmful labelling.