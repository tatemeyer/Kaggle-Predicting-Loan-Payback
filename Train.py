import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")
sample_submission = pd.read_csv("dataset/sample_submission.csv")

print("Train head:")
print(train.head())
print("\nTest head:")
print(test.head())
print("\nSample submission head:")
print(sample_submission.head())

TARGET_COLUMN = "loan_paid_back"

catagorical_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

numerical_features = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate"
]

feature_columns = numerical_features + catagorical_features

X = train[feature_columns]
y = train[TARGET_COLUMN]

X_test_final = test[feature_columns]


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=catagorical_features
)

valid_pool = Pool(
    data=X_valid,
    label=y_valid,
    cat_features=catagorical_features
)

full_train_pool = Pool(
    data=X,
    label=y,
    cat_features=catagorical_features
)

test_pool = Pool(
    data=X_test_final,
    cat_features=catagorical_features
)

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=200,
)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
    early_stopping_rounds=100
)

y_valid_proba = model.predict_proba(X_valid)[:, 1]
y_valid_pred_05 = (y_valid_proba >= 0.5).astype(int)

auc = roc_auc_score(y_valid, y_valid_proba)
print(f"\nValidation AUC: {auc:.4f}")

print("\nClassification report (threshold = 0.50):")
print(classification_report(y_valid, y_valid_pred_05))

print("Confusion matrix (threshold = 0.50):")
print(confusion_matrix(y_valid, y_valid_pred_05))


best_thr = 0.5
best_f1 = 0.0

thresholds = np.linspace(0.1,0.9,17)

for thr in thresholds:
    preds = (y_valid_proba >= thr).astype(int)
    f1 = f1_score(y_valid, preds, pos_label=1.0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
    
print(f"\nBest threshold for class 1 F1: {best_thr: .2f} (F1={best_f1: .4f})")

y_valid_pred_best = (y_valid_proba >= best_thr).astype(int)

print("\nClassification report (best threshold):")
print(classification_report(y_valid, y_valid_pred_best))

print("Confusion matrix (best threshold):")
print(confusion_matrix(y_valid, y_valid_pred_best))

final_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=model.tree_count_,  # reuse best iteration found above
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=200,
)

final_model.fit(
    full_train_pool,
    use_best_model=False,
)

test_proba = final_model.predict_proba(test_pool)[:, 1]  # P(loan_paid_back = 1)

submission = sample_submission.copy()
submission["loan_paid_back"] = test_proba  # PROBABILITIES, not class labels

output_path = "submission_catboost_proba.csv"
submission.to_csv(output_path, index=False)

print(f"\nSaved probability submission to: {output_path}")
print(submission.head())   