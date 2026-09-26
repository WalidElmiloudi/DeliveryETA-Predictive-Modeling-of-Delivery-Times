import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../data/processed/featured.csv")

X = df.drop("Time_taken(min)",axis=1)
y = df["Time_taken(min)"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numerical_features = X_train.select_dtypes(include=["Int64","Float64"]).columns
categorical_features = X_train.select_dtypes(exclude=["Int64","Float64"]).columns

numerical_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("impute",SimpleImputer(strategy="most_frequent")),
    ("scaler",OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_features),
    ("cat",categorical_pipeline,categorical_features)
])

random_forest_model = Pipeline([
    ("preprocessor",preprocessor),
    ("regressor",RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='sqrt'
    ))
])

random_forest_model.fit(X_train,y_train)