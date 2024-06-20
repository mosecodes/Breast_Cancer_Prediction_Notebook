import pandas as pd
import numpy as np
import preprocessor as prep

from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# set pipelines to use pandas dataframes
# -- will maintain column names for better displays
set_config(transform_output="pandas")


# helper func for returning boolean list for feature selection
def x_feature_list(data):
    result = []
    for element in data:
        if element != 'S/N' and element != 'Diagnosis Result':
            result.append(element)
    return result


# CREATE DATAFRAME FROM CSV FILE
df_dtypes = {'S/N': np.int64, 'Year': pd.Int64Dtype(), 'Age': pd.Int64Dtype(), 'Menopause': np.int64,
             'Tumor Size (cm)': pd.Int64Dtype(), 'Inv-Nodes': pd.Int64Dtype(),
             'Breast': "category", 'Metastasis': pd.Int64Dtype(),
             'Breast Quadrant': "category", 'History': pd.Int64Dtype(),
             'Diagnosis Result': "category"}

df = pd.read_csv("breast-cancer-dataset.csv",
                 na_values="#",
                 dtype=df_dtypes,
                 index_col='S/N')
print('\nRAW DATAFRAME EXTRACTED FROM CSV FILE')
df.info()
print("\nREMOVING ALL ROWS WITH NULL STRING FIELDS")
try:
    cols_to_check = ["Year", "Menopause", "Inv-Nodes", "Metastasis", "History", "Breast", "Breast Quadrant"]
    df.dropna(subset=cols_to_check, inplace=True)
    print("\nREMOVED NULL STRING ROWS, NEW DATAFRAME INFO BELOW:\n")
    df.info()

except TypeError:
    print('Exception when removing null data')

# SETTING DATA AND TARGET
print('\nSETTING DATA AND TARGET')
X, y = (df.filter(x_feature_list(df.columns))), (df.filter(["Diagnosis Result"]).map(lambda x: 1 if x == 'Malignant'
else 0, na_action='ignore').squeeze())
# encode target variables
# VALIDATE X, y
print('\nVALIDATING X\n')
X.info()
print('\nVALIDATING y\n')
y.info()
print('\nshape of y:')
print(y.shape)


# CALL PREPROCESSOR
preprocessor = prep.preprocess()
print('Preprocessor called\n')

# CLASSIFIER
clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())],
    verbose=True,
)
print('\nCLASSIFIER PIPELINE BUILT\n')

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("Data successfully split\n")

# RUN DATA THROUGH PREPROCESSOR
# print('FITTING TRAINING DATA TO PREPROCESSOR')
# try:
#     preprocessor.fit_transform(X_train, y_train)
# except Exception as e:
#     print(e)

# FIT DATA TO CLASSIFIER
clf.fit(X_train, y_train)
print('CLASSIFIER READY: CLASSES SHOWN BELOW')
print(clf.classes_)
print('\nCLASSIFIER ACCURACY SCORE:')
print(accuracy_score(clf.predict(X_test), y_test))
