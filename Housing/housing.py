# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

housing = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
housing.info()
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
