#%% 
import pandas as pd
import altair as alt
#Install altair_data_server to handle large dataset
alt.data_transformers.enable('data_server')
#Load campaign dataset
campaign = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
campaign.info() 

#%%
# Luke's Cells
import seaborn as sns
# correlation matrix helps us see what features are related and by how much. 
corr_matrix = campaign.corr()

# visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True)

# euribor3m has high correlation with a number of other features
# previous is also a good one too

#%%
from sklearn.model_selection import train_test_split
from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
#clf = KNeighborsClassifier(n_neighbors=3)
features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = pd.get_dummies(campaign[features])
y = campaign['y']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 420)
#clf.fit(x_train, y_train)
#pred = clf.predict(x_test)
#pred
#accuracy_score(y_test, pred) #0.885

# Build the decision tree
clf = DecisionTreeClassifier()

# Train it
clf.fit(x_train, y_train)

# Test it 
clf.score(x_test, y_test)

# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        cmap=plt.cm.Blues
    )

# %%
from imblearn.over_sampling import RandomOverSampler
ro = RandomOverSampler()
features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = campaign[features]
y = campaign['y']

# Oversample, note that we oversample X and y at the same time in order to 
# make sure our features and targets stay synched.
X_new, y_new = ro.fit_resample(X, y)

# Convert this to a dataframe and check the counts, now they're equal, because
# we have a bunch of duplicate survivors
customer = pd.DataFrame(y_new)
customer.value_counts()


# %%
x_train, x_test, y_train, y_test = train_test_split(X_new, y_new, test_size = 0.2, random_state = 420)
# Build the decision tree
clf = DecisionTreeClassifier()

# Train it
clf.fit(x_train, y_train)

# Test it 
clf.score(x_test, y_test)
# %%
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        cmap=plt.cm.Blues
    )

# %%