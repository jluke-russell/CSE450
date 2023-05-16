# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
housing.info()

housing['total_property_sqft'] = housing['sqft_lot'] + housing['sqft_living']

housing['total_property_sqft_dif_of_neighbors'] = (housing['sqft_lot'] + housing['sqft_living']) - (housing['sqft_living15'] + housing['sqft_lot15'])

housing['total_neighbor_property_sqft;'] = housing['sqft_living15'] + housing['sqft_lot15']
# %%
import seaborn as sns
# correlation matrix helps us see what features are related and by how much. 
corr_matrix = housing.corr()
# visualize the correlation matrix
fig, ax = plt.subplots(figsize=(20, 18))
hm = sns.heatmap(corr_matrix, annot=True)
hm.set_title("Correlation Heatmap of Housing Data")
plt.show()


# %%
#average housing price: 539436.71295
average_price = housing['price'].mean()
print(average_price)

# Get target variable and features and split them into test and train datasets
X = housing[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade',
         'sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15','total_property_sqft',
        'total_property_sqft_dif_of_neighbors','total_neighbor_property_sqft;']]
y = housing['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model and train it
model = XGBRegressor()
model.fit(X_train, y_train)

# Get predictions for test data
predictions = model.predict(X_test)
predictions

# Compute the Root Mean Squared Error of the predictions
from sklearn.metrics import mean_squared_error

result = mean_squared_error(y_test, predictions, squared=False)
result #we're off by 197699 price on average