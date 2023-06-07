# %%
import pandas as pd
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

bikes = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bikes.csv")

bikes.head()
# %% 
# Make actual target 
bikes['total'] = bikes['casual'] + bikes['registered']
bikes.head()

# %%
# Create features



# %%
X = bikes[['season','hr','holiday','workingday','weathersit','hum','windspeed','temp_c','feels_like_c']]
y = bikes['total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 420)

# Scale the data
minmax_scaler = preprocessing.MinMaxScaler()
X_train = minmax_scaler.fit_transform(X_train) # fit the scale to the training data
X_test = minmax_scaler.transform(X_test) # use the same scale on the testing data

# %%
# Initialize the Neural Network
model = Sequential() # Sequential just means the network doesn't have loops--the outputs of one layer of neurons go to the next layer of neurons

# Add the first layer
model.add(Dense(16, input_dim=9, activation='relu')) # This layer has 16 neurons. They are each connected (dense) to the input neurons.
# Note: We need the input dimension to match the number of features at our input layer

# Add another "hidden layer"
model.add(Dense(8, activation = 'relu')) # This layer has 8 neurons

# Add the "output layer"
model.add(Dense(1, activation='linear')) # Our last layer doesn't need a non-linear activation function, unless it is useful for the type of answer we want
# The ouput layer should have the same number of neurons as outputs you are generating. In this case, it is just producing one number. 

# Compile model
model.compile(loss='MSE', optimizer= 'Adam', metrics=['mean_squared_error'])

# %%
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 300, verbose = 0)

# Evaluate the model on the training data
_, train_mse = model.evaluate(X_train, y_train, verbose = 1)

# Evaluate the model on the testing data
_, test_mse = model.evaluate(X_test, y_test, verbose = 1)

# Get predictions for the testing data
predictions = model.predict(X_test)

# Get the r^2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print(r2)
# %%
