import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

##read dataset

df = pd.read_csv('../../data/raw/housing.csv')
df.head()

## checking distributions for dataset

feat_numeric = list(df.select_dtypes(include=['number']).columns)
plt.figure(figsize=(20,15))
plt.suptitle('numerical feature distribution',fontsize=25)
for i in range(0,len(feat_numeric)):
    plt.subplot(3,3,i+1)
    sns.histplot(df[feat_numeric[i]],fill=True,color='b',kde=True)
plt.tight_layout()
plt.savefig('../../reports/figures/Distributions/num_feat_distribtion.png')
plt.show()
plt.close()

"""
Most distributions are tail-heavy so we may need transformation

"""
####### Binning to convert from numerical to catagories 
"""
from expert domain we are sure that the median_income is an 
important attribute for predicting median housing prices so 
we need to make sure my test data is representative of the various
categories of income in whole dataset
"""

df['income_cat'] = pd.cut(df['median_income'],
                          bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                          labels=[1, 2, 3, 4, 5])
df['income_cat'].hist()

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for  train_index, test_index in split.split(df,df['income_cat']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
    
##check if the spilt is stratified or not

strat_test_set['income_cat'].value_counts()/len(strat_test_set)
strat_train_set['income_cat'].value_counts()/len(strat_train_set)

##drop the cat income column
strat_train_set.drop(['income_cat'],axis=1,inplace=True)
strat_test_set.drop(['income_cat'],axis=1,inplace=True)



##check the patterns in data

plt.figure(figsize=(15,20))
plt.suptitle('scatter plots for numerical features')
for i in range(0,len(feat_numeric)):
    plt.subplot(5,2,i+1)
    plt.scatter(y=feat_numeric[i],x=strat_train_set.index,data=strat_train_set,color='r')
    plt.ylabel(feat_numeric[i])
    plt.tight_layout()
plt.savefig(f'../../reports/figures/patterns_in_data.png')
plt.show()
plt.close()

##visualize Geographical data
strat_train_set.plot(kind='scatter',x="longitude",y="latitude"
            ,alpha=0.4,c=strat_train_set.median_house_value
            ,s=strat_train_set['population']/100,label='Population'
            ,cmap=plt.get_cmap('jet'),colorbar=True)
plt.savefig(f'../../reports/figures/Geograhical_data.png')
plt.show()
plt.close()


#Check linear corr

plt.figure(figsize=(15,15))
sns.heatmap(strat_train_set[feat_numeric].corr(),annot=True)
plt.savefig(f'../../reports/figures/corr_matrix.png')
plt.show()
plt.close()

## median_income vs median_house_value

strat_train_set.plot(kind='scatter',x='median_income'
                     ,y='median_house_value',alpha=0.1)
"""
you can see some data quirks as straight lines

"""

##export dataset
strat_train_set.to_pickle('../../data/processed/train_set.pkl')
strat_test_set.to_pickle('../../data/processed/test_set.pkl')