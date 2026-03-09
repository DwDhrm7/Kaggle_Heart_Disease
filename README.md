# ❤️ Heart Disease Prediction -- Kaggle Competition

Machine Learning project for predicting the presence of heart disease
using structured clinical data from the Kaggle Playground Series
dataset.

## 📌 Overview

This project builds a machine learning model to classify whether a
patient has **heart disease (Presence)** or **no heart disease
(Absence)**.

The solution applies:

-   Feature engineering
-   Stratified cross-validation
-   Gradient boosting model (**LightGBM**)

The evaluation metric used in the competition is **ROC-AUC**.

------------------------------------------------------------------------

# 📊 Dataset

Dataset source: **Kaggle Playground Series -- Season 6 Episode 2**

The dataset contains medical attributes such as:

  Feature                   Description
  ------------------------- ------------------------------------------
  Age                       Patient age
  Sex                       Gender
  Chest pain type           Type of chest pain
  Chol                      Cholesterol level
  FBS over 120              Fasting blood sugar indicator
  EKG results               Electrocardiogram results
  Max HR                    Maximum heart rate
  Exercise angina           Exercise induced angina
  ST depression             ST depression value
  Number of vessels fluro   Number of vessels colored by fluoroscopy
  Thallium                  Thallium stress test result

Target variable:

Heart Disease\
0 = Absence\
1 = Presence

------------------------------------------------------------------------

# 🧠 Machine Learning Approach

## 1️⃣ Data Preprocessing

The target label is converted into numerical format:

``` python
train["Heart Disease"] = train["Heart Disease"].map({
    "Absence": 0,
    "Presence": 1
})
```

------------------------------------------------------------------------

## 2️⃣ Feature Engineering

Additional features were created to improve model performance.

### Age Binning

``` python
df['Age_bin'] = pd.cut(
    df['Age'],
    bins=[0,40,50,60,70,100],
    labels=[0,1,2,3,4]
).astype('int')
```

### Cholesterol Binning

Cholesterol values were grouped into bins to capture risk categories.

### ST Depression Severity

ST depression values were transformed into severity levels.

These transformations help the model capture **non-linear
relationships**.

------------------------------------------------------------------------

# 🤖 Model

The model used in this project is **LightGBM**, a gradient boosting
framework optimized for speed and performance.

### Model Parameters

``` python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.04,
    'num_leaves': 150,
    'max_depth': 10,
    'min_child_samples': 40,
    'subsample': 0.85,
    'colsample_bytree': 0.8
}
```

Why LightGBM?

-   Very fast training
-   Handles categorical features
-   Excellent performance on tabular datasets

------------------------------------------------------------------------

# 🔁 Cross Validation

The model uses **Stratified K-Fold Cross Validation**.

Benefits:

-   Maintains class distribution
-   More reliable performance estimation
-   Reduces overfitting risk

Evaluation metric:

ROC-AUC Score

------------------------------------------------------------------------

# 📦 Final Training

After cross-validation, the final model is trained on the **full
dataset**.

``` python
full_data = lgb.Dataset(X, label=y, categorical_feature=cat_cols)

final = lgb.train(
    params,
    full_data,
    num_boost_round=int(model.best_iteration * 1.1)
)
```

------------------------------------------------------------------------

# 📈 Prediction

Predictions are generated for the test dataset:

``` python
pred = final.predict(X_test)
```

Submission file format:

``` python
sub = pd.DataFrame({
    "id": test["id"],
    "Heart Disease": pred
})
```

------------------------------------------------------------------------

# 🛠️ Requirements

Install dependencies before running the notebook.

``` bash
pip install pandas
pip install numpy
pip install lightgbm
pip install scikit-learn
```

------------------------------------------------------------------------

# 📁 Project Structure

    heart-disease-kaggle/
    │
    ├── heart_disease_model.ipynb
    ├── README.md
    └── submission.csv

------------------------------------------------------------------------

# 🚀 How to Run

1.  Download dataset from Kaggle
2.  Open the notebook
3.  Run all cells

``` bash
jupyter notebook
```

or run on **Kaggle Notebook / Google Colab**.

------------------------------------------------------------------------

# 🎯 Future Improvements

Possible improvements for better leaderboard performance:

-   Hyperparameter tuning
-   Feature interaction
-   Ensemble models
-   AutoML approaches
-   Feature selection techniques

------------------------------------------------------------------------

# 👨‍💻 Author

**I Dewa Made Dharma Putra Santika**\
AI • Robotics • Machine Learning Enthusiast
