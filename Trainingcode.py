#importing library
import pandas as pd

#importing dataset
#df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
df = pd.read_csv("Melbourne_housing_FULL.csv")
from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.metrics import mean_absolute_error

##previewdataframe


del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']
df.dropna(axis=0, how='any', subset= None, inplace = True)

df = pd.get_dummies(df, columns = ['Suburb', 'CouncilArea', 'Type'])

# setting price as the dependent variable, all else independent
X = df.drop('Price', axis=1)
y = df['Price']

#Split dataset randomly between training and test:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, shuffle = True)

#Select algorithm and hyperparameters: 
model = ensemble.GradientBoostingRegressor(
    n_estimators = 250,
    learning_rate = 0.1,
    max_depth = 5,
    min_samples_split = 10,
    min_samples_leaf = 6,
    max_features = 0.6,
    loss = 'huber' 
)
#train model using fit()
model.fit(X_train, y_train)

#Evaluate results using the training data
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.2f" % mae_train)

#Evaluate results using the test data
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.2f" % mae_test)

df.head(30)