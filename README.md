## Building an Machine Learning model for predicting Heart Disease using Classification
* Goal: Predict whether a patient has heart disease (target = 1) or not (target = 0) using the patients medical attributes.
### Worked with a structural dataset(thanks to kaggle) containing 303 x 14 (rows & columns):
* age - age in years
* sex - 1 = male, 0 = female
* cp - chest pain type (0–3)
* trestbps - resting blood pressure
* chol - serum cholesterol
* fbs - fasting blood sugar > 120 mg/dl
* restecg - resting electrocardiographic results
* thalach - maximum heart rate achieved
* exang - exercise-induced angina
* oldpeak - ST depression induced by exercise
* slope - slope of peak exercise ST segment
* ca - number of major vessels (0–3)
* thal - thalassemia
* target - (1 = heart disease, 0 = no disease)
### Workflow Approach
* Problem
* Data
* Evaluation
* Features
* Modelling
* Experimentation
### Steps included 
#### Imported tools & libraries
* Data & plotting: pandas, numpy, matplotlib, seaborn
* ML models: LogisticRegression, KNeighborsClassifier, RandomForestClassifier
* Model selection: train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
* Metrics: confusion_matrix, classification_report, precision_score, recall_score, f1_score, RocCurveDisplay
* Persistence: joblib (dump, load)
#### Loaded and inspected the data
* df = pd.read_csv("heart-disease.csv")
* Checked shape ((303, 14)), .head(), .tail()
* Looked at class distribution:target = 1 (disease): 165, target = 0 (no disease): 138
* Verified missing values: df.isna().sum() → no problematic NaNs
#### Exploratory Data Analysis (EDA) & visualization
* Target balance : df["target"].value_counts() and bar plot
* Sex vs heart disease : pd.crosstab(df.target, df.sex) and bar plot
  - Showed different frequencies of disease between males and females.
* Age & max heart rate vs disease
* Scatter plot: age vs thalach, colored by target
  - Visualized how patients with and without heart disease are distributed across age and max heart rate.
* Chest pain type vs disease
  - pd.crosstab(df.cp, df.target) and bar plot
    - Showed certain chest pain types associated more with heart disease.
#### Correlation analysis
* df.corr() and a seaborn heatmap
  - Identified which features correlate with target (e.g., cp, thalach, oldpeak, ca, thal, etc.)
#### Prepared features and labels
* Features: X = df.drop("target", axis=1)
* Target: y = df["target"]
* Split data to train and test.
#### Built baseline models
* Created a dictionary of models, which included:
* LogisticRegression()
* KNeighborsClassifier()
* RandomForestClassifier()
#### Wrote fit_and_score() to:
* Fit each model on X_train, y_train
* Evaluate .score(X_test, y_test) (accuracy)
* Compared accuracies in a bar chart.
* With the dataset and split I used, the test accuracies end up around:
  - Logistic Regression: ~0.85
  - Random Forest: ~0.84
  - KNN: ~0.70
#### Hypertuned KNN
* Manually tried n_neighbors from 1 to 30, and received the KNN test accuracy of ~0.74
* Plotted train vs test accuracy as a function of n_neighbors.
* As KNN test score is lower than Logistic Regression, so it was not chosen as final model.
#### Hyperparameter tuning with RandomizedSearchCV (Random Forest)
* Defined a random forest hyperparameter grid.
* Used RandomizedSearchCV(RandomForestClassifier(), param_distributions=rf_grid, cv=5, n_iter=20)
* Fitted and checked rs_rf.best_params_ and test score
* RF after tuning got test accuracy similar to Logistic Regression (~0.85), but I continued with Logistic Regression as your main final model (better interpretability and simpler).
#### Hyperparameter tuning with GridSearchCV (Logistic Regression)
* Defined a Logistic Regression hyperparameter grid.
* Used GridSearchCV(LogisticRegression(),param_grid=log_reg_grid,cv=5,verbose=True)
* Then fitted gs_log_reg.fit(X_train, y_train)
* Next, retrieved best parameters:gs_log_reg.best_params_
* Checked Test accuracy of tuned logistic model
* Got score of gs_log_reg.score 0.85
* Evaluation of the tuned model Predictions (y_preds = gs_log_reg.predict(X_test))
  - Metrics:
      - Accuracy: ~0.85
  - From my classification_report:
      - For class 1 (heart disease):
      - Precision ≈ 0.82
      - Recall ≈ 0.97
      - F1-score ≈ 0.89
      - Support: 61 test samples total
  - Confusion matrix and seaborn heatmap:
      - True negatives: 15
      - False positives: 8
      - False negatives: 1
      - True positives: 37
  - ROC curve using RocCurveDisplay.from_estimator(gs_log_reg, X_test, y_test)
#### Cross-validated metrics with the “best” Logistic Regression
* Created clf = LogisticRegression(C=0.38566, solver="liblinear")
* Used 5-fold cross-validation over the full dataset:
  - Mean CV Accuracy: ~0.84
  - Mean CV Precision: ~0.82
  - Mean CV Recall: ~0.92
  - Mean CV F1: ~0.86
* Plotted these metrics as a bar chart.
#### Model interpretability – feature importance
* Used logistic regression coefficients.
* Found that features like sex, thal, cp, ca, oldpeak have higher absolute coefficients, meaning they’re more influential in the prediction.
#### Model saving and re-loading
* Saved tuned model with joblib using:
  - from joblib import dump, load
* Loaded it back and checked predictions
* Confirmed the loaded model gives the same performance.
### Core ML steps you performed
  - I have touched all the main steps of a supervised ML project.
* Problem definition - Binary classification: predict heart disease (yes/no).
* Data loading - Read CSV using pandas.
* Data exploration & sanity checks - Shape, head, distribution, missing values.
* EDA & domain insights - Target balance, sex distribution, chest pain types, age vs heart rate, correlations.
* Feature/label separation - X and y.
* Train–test split - Hold-out evaluation for generalization.
* Baseline modeling - Logistic Regression, KNN, Random Forest.
* Model comparison - Compare accuracy across models.
* Hyperparameter tuning - KNN (neighbors), RandomForest (RandomizedSearch), LogisticRegression (GridSearch).
* Evaluation with multiple metrics - Accuracy, precision, recall, F1, confusion matrix, ROC curve.
* Cross-validation - More robust metrics via cross_val_score.
* Model interpretability - Coefficient-based feature importance.
* Model persistence - Save and load model with joblib.
### Acheived results
* I have ended up with a tuned Logistic Regression model.
* Achieves about 86% accuracy on the test set.
* Has high recall (~0.97) for the heart disease class (which is I believe good for medical problems where missing a positive case is costly).
* Balanced precision and F1-score for the positive class (~0.82 precision, ~0.89 F1)
* Generalizes reasonably well according to 5-fold cross-validation metrics.
* Acheived all these results by performing hypertuning, comparing multiple algorithms etc.
### Key Learnings
* During this project, I learned how to:
  - Perform Exploratory Data Analysis (EDA) on a real-world health dataset.
  - Build and compare multiple classification models (Logistic Regression, KNN, Random Forest).
  - Use hyperparameter tuning techniques.
  - Manual parameter sweeps (e.g., KNN neighbors)
  - RandomizedSearchCV and GridSearchCV
  - Evaluate models with multiple metrics beyond accuracy:
  - Precision, recall, F1-score, confusion matrix, ROC curve
  - Use cross-validation to obtain more reliable performance estimates.
  - Interpret Logistic Regression coefficients to understand feature importance.
  - Save and load machine learning models using joblib for future use.
#### This is just a practice project where I have used the skills that i have learned so far. I believe there can be differnt ways to tune a hyperparameter which may lead to better accuracy ( or either Recall score/F1 score).


