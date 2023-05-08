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
hm = sns.heatmap(corr_matrix, annot=True)
hm.set_title("Correlation Heatmap of Campaign Data")

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
features = ['euribor3m', 'nr.employed']
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
campaign = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
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
from sklearn import metrics
predictions = clf.predict(x_test)
print(metrics.confusion_matrix(y_test, predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# %%
import pandas as pd
import matplotlib.pyplot as plt

campaign = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

ro = RandomOverSampler()
features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = campaign[features]
y = campaign['y']

# Oversample, note that we oversample X and y at the same time in order to 
# make sure our features and targets stay synched.
X_newer, y_newer = ro.fit_resample(X, y)

# Convert this to a dataframe and check the counts, now they're equal, because
# we have a bunch of duplicate survivors
customer = pd.DataFrame(y_newer)
customer.value_counts()

clf = RandomForestClassifier()

features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = pd.get_dummies(campaign[features])
y = campaign['y']
x_train, x_test, y_train, y_test = train_test_split(X_newer, y_newer, test_size = 0.2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
predictions = clf.predict(x_test)
print(classification_report(y_test,predictions))

ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        cmap=plt.cm.Blues
    )


#%%
import altair as alt 
# explore other features and chatacteristics of people 

# create the scatter plot
scatter = alt.Chart(campaign).mark_bar().encode(
    x='month',
    y='nr.employed',
    color = 'y'
)

# show the plot
scatter






# %%
import pandas as pd
import matplotlib.pyplot as plt

my_test = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank_holdout_test_mini.csv')

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

ro = RandomOverSampler()
features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = my_test[features]

# Oversample, note that we oversample X and y at the same time in order to 
# make sure our features and targets stay synched.
X_newer, y_newer = ro.fit_resample(X, y)

# Convert this to a dataframe and check the counts, now they're equal, because
# we have a bunch of duplicate survivors
customer = pd.DataFrame(y_newer)
customer.value_counts()

clf = RandomForestClassifier()

features = ['euribor3m', 'emp.var.rate', 'nr.employed', 'cons.price.idx']
X = pd.get_dummies(my_test[features])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_pred)
predictions = clf.predict(x_test)
print(classification_report(y_test,predictions))

ConfusionMatrixDisplay.from_estimator(
        clf,
        x_test,
        y_test,
        cmap=plt.cm.Blues
    )
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

data = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

def fix_data(data, drop_na, features, allFeatures):
    # remove unwanted features
    for feature in allFeatures:
        if feature not in features:
            data = data.drop(feature, axis=1)

    pd.set_option('display.max_columns', None)
    # print(topRows)

    # replace unknown to nan
    if drop_na:
        data = data.replace("unknown", np.nan)
        data = data.dropna()

    if 'day_of_week' in features:
        data['day_of_week'] = data['day_of_week'].replace(['mon', 'tue', 'wed', 'thu', 'fri'], [1, 2, 3, 4, 5])

    # replace default to 0 and 1
    if 'default' in features:
        data['default'] = data['default'].replace({'unknown': 0, 'no': 1, 'yes': 2})

    # replace housing
    if 'housing' in features:
        data['housing'] = data['housing'].replace({'unknown': 0, 'yes': 1, 'no': 0})

    # replace contact
    if 'contact' in features:
        data['contact'] = data['contact'].replace({'unknown': 0, 'cellular': 1, 'telephone': 0})

    # replace education with 0, 1, 2, 3
    if 'education' in features:
        data['education'] = data['education'].replace('unknown', 0)
        data['education'] = data['education'].replace('illiterate', 1)
        data['education'] = data['education'].replace('basic.4y', 2)
        data['education'] = data['education'].replace('basic.6y', 3)
        data['education'] = data['education'].replace('basic.9y', 4)
        data['education'] = data['education'].replace('high.school', 5)
        data['education'] = data['education'].replace('professional.course', 6)
        data['education'] = data['education'].replace('university.degree', 7)

    # replace marital with 0, 1, 2
    if 'marital' in features:
        data['marital'] = data['marital'].replace('unknown', 0)
        data['marital'] = data['marital'].replace('single', 1)
        data['marital'] = data['marital'].replace('married', 2)
        data['marital'] = data['marital'].replace('divorced', 3)

    # replace loan with 0, 1
    if 'loan' in features:
        data['loan'] = data['loan'].replace('unknown', 0)
        data['loan'] = data['loan'].replace('yes', 1)
        data['loan'] = data['loan'].replace('no', 2)

    # replace poutcome with 0, 1, 2
    if 'poutcome' in features:
        data['poutcome'] = data['poutcome'].replace('unknown', 0)
        data['poutcome'] = data['poutcome'].replace('nonexistent', 1)
        data['poutcome'] = data['poutcome'].replace('failure', 2)
        data['poutcome'] = data['poutcome'].replace('success', 3)

    # replace job with 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    if 'job' in features:
        data['job'] = data['job'].replace('unknown', 0)
        data['job'] = data['job'].replace('admin.', 1)
        data['job'] = data['job'].replace('blue-collar', 2)
        data['job'] = data['job'].replace('entrepreneur', 3)
        data['job'] = data['job'].replace('housemaid', 4)
        data['job'] = data['job'].replace('management', 5)
        data['job'] = data['job'].replace('retired', 6)
        data['job'] = data['job'].replace('self-employed', 7)
        data['job'] = data['job'].replace('services', 8)
        data['job'] = data['job'].replace('student', 9)
        data['job'] = data['job'].replace('technician', 10)
        data['job'] = data['job'].replace('unemployed', 11)

    # replace month with 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    if 'month' in features:
        data['month'] = data['month'].replace('unknown', 0)
        data['month'] = data['month'].replace('jan', 1)
        data['month'] = data['month'].replace('feb', 2)
        data['month'] = data['month'].replace('mar', 3)
        data['month'] = data['month'].replace('apr', 4)
        data['month'] = data['month'].replace('may', 5)
        data['month'] = data['month'].replace('jun', 6)
        data['month'] = data['month'].replace('jul', 7)
        data['month'] = data['month'].replace('aug', 8)
        data['month'] = data['month'].replace('sep', 9)
        data['month'] = data['month'].replace('oct', 10)
        data['month'] = data['month'].replace('nov', 11)
        data['month'] = data['month'].replace('dec', 12)

    # replace day_of_week with 0, 1, 2, 3, 4
    if 'dayofweek' in features:
        data['day_of_week'] = data['day_of_week'].replace('unknown', 0)
        data['day_of_week'] = data['day_of_week'].replace('mon', 1)
        data['day_of_week'] = data['day_of_week'].replace('tue', 2)
        data['day_of_week'] = data['day_of_week'].replace('wed', 3)
        data['day_of_week'] = data['day_of_week'].replace('thu', 4)
        data['day_of_week'] = data['day_of_week'].replace('fri', 5)

    return data

# bank is our OG dataset; don't mess with it
bank = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/bank.csv')

allFeatures = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
               'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
               'euribor3m', 'nr.employed']

features = ['age', 'job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week',
            'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
            'nr.employed']

bank = fix_data(bank, False, features, allFeatures)

X = bank[features]
y = bank['y']

ro = RandomOverSampler()

X_new, y_new = ro.fit_resample(X, y)
X = X_new
y = y_new

# Split our data into training and test data, with 30% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Build the decision tree
# clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
# random forest

clf = RandomForestClassifier(max_depth=30, random_state=42, criterion='entropy', n_estimators=100)

clf.fit(X_train, y_train)

# Test it
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues
    )

# %%
import seaborn as sns
# correlation matrix helps us see what features are related and by how much. 
corr_matrix = bank.corr()

# visualize the correlation matrix
fig, ax = plt.subplots(figsize=(20, 18))
hm = sns.heatmap(corr_matrix, annot=True)
hm.set_title("Correlation Heatmap of Campaign Data")
plt.show()
# %%
