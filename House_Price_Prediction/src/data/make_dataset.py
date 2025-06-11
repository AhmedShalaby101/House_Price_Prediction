import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##read dataset

df = pd.read_csv('../../data/raw/housing.csv')
df.head()

df.info()
"""
 All attributes are numerical except ocean_promixity which will need encoding
"""

##checking missing values

df.isnull().sum()
"""
 total_bedrooms       207 missing

"""

##check categories for catagorical feats

df['ocean_proximity'].unique()
op =df.groupby(['ocean_proximity'])['median_house_value'].mean()
op.sort_values(ascending=False)
""" 
   we use OHE to encode these 5 categories 
"""






