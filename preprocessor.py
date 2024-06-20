from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.compose import ColumnTransformer


def preprocess():
    # PREPROCESSING
    # numeric features
    numeric_features = ["Age", "Tumor Size (cm)"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    # categorical features
    categorical_features = ["Year", "Menopause", "Inv-Nodes", "Metastasis", "History", "Breast", "Breast Quadrant"]
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("selector", SelectPercentile(chi2, percentile=80)),
        ],
        verbose=False
    )
    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
