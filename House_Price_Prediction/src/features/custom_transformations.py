import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

## Construct features that we thought it will be useful

rooms_ix , bedrooms_ix , population_ix , households_ix = 3,4,5,6

class AttributesAdder(BaseEstimator,TransformerMixin):
    
    ## add_bedrooms_per_room works here as a hyperparameter 
    
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    ##fit and transform functions must be in the class to works 
    ## will with sklearn piplines
    
    def fit(self,X,Y=None):
        return self 
    def transform(self,X):
        rooms_per_household = X[:,rooms_ix] / X[:,households_ix]
        population_per_household = X[:,population_ix] / X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms = X[:,bedrooms_ix] / X[:,rooms_ix]
            return np.c_[
                X,rooms_per_household,population_per_household,bedrooms_per_rooms
            ]
        else:
            return np.c_[
                X,rooms_per_household,population_per_household
            ]
 