import pandas as pd
from sklearn.impute import SimpleImputer
from custom_transformations import AttributesAdder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
##read dataset

df = pd.read_pickle('../../data/processed/train_set.pkl')

X_train  = df.drop(['median_house_value'],axis=1)
Y_train  = df['median_house_value']
numeric_feat = X_train.select_dtypes(include="number").columns.tolist()
cat_feat = X_train.select_dtypes(include='object').columns.tolist()

##pipline for numerical features

num_pipline = Pipeline([
    ("impute_missing_values", SimpleImputer(strategy='median')),
    ("add_features", AttributesAdder()),
    ('scaling_data', StandardScaler())
])

## full pipline 

full_pipline = ColumnTransformer([
    ('numerical',num_pipline,numeric_feat),
    ('catagorical',OneHotEncoder(),cat_feat)
])


X_train = pd.DataFrame(full_pipline.fit_transform(X_train))

## export processed dataset
X_train.to_pickle('../../data/interim/model_X_train.pkl')
Y_train.to_pickle('../../data/interim/model_Y_train.pkl')


        