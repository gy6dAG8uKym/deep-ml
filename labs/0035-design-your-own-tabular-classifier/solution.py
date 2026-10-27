# --- cell 1 ---
import numpy as np
import pandas as pd
# You may import numpy, pandas, scipy, and sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer


def train(X_train, y_train, X_val, y_val):
    """
    Design your own tabular binary classifier on real census data.

    X_* are pandas DataFrames with MIXED dtypes (numeric + categorical).
    Features are RAW — you own imputation, encoding, scaling, imbalance
    handling, model choice, and calibration.

    Target: predict whether income is >50K (class 1, minority).

    Args:
        X_train: pd.DataFrame (n_train, n_features)
        y_train: np.ndarray (n_train,) binary labels {0, 1}
        X_val:   pd.DataFrame (n_val, n_features)
        y_val:   np.ndarray (n_val,) binary labels {0, 1}

    Returns:
        predict_proba: callable predict_proba(X) -> np.ndarray (n,)
            Positive-class scores for a DataFrame X with the same
            columns as X_train. Finite scores; ranking is what matters
            for PR-AUC.
    """
    # TODO: build any classical pipeline you like
    # Ideas:
    #   - ColumnTransformer: impute + scale numerics, encode categoricals
    #   - class_weight, resampling / SMOTE on the train fold only
    #   - logistic / SVM / RF / HistGradientBoosting / stacking
    #   - tune on val, refit on train+val
    #   - calibrate on val
    numeric_cols = []
    categorical_cols =[]
    
    X_train=X_train.copy()
    
    for col in X_train.columns:
        converted = pd.to_numeric(
            X_train[col],
            errors="coerce"
        )

        if converted.notna().sum() == X_train[col].notna().sum():
            numeric_cols.append(col)
            X_train[col] = converted
        else:
            categorical_cols.append(col)
        
        
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
        ])
        
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
        ])
    
    model.fit(X_train,y_train)
    

    def predict_proba(X):
        X=X.copy()
        
        for col in numeric_cols:
            X[col]=pd.to_numeric(X[col],errors="coerce")
            
        return model.predict_proba(X)[:,1]

    return predict_proba