#!/usr/bin/env python
# coding: utf-8

# # Data Analysis and preparation

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('wifi_data_rev2.csv')


# Checking to see what the data looks like.

# In[3]:


df.head()


# In[4]:


print('Shape of the dataframe ', df.shape)
#print(df.describe())
df.isnull().sum()


# In[5]:


# preparing the inputs and outputs
# dropping rows where either 'x' or 'y' is NaN
df_cleaned = df.dropna(subset=['x', 'y'])

print('Shape of cleaned dataframe ', df_cleaned.shape)
df_cleaned.isnull().sum()


# **23 rows where dropped** as a result of this action.

# In[6]:


# y_reg is for the regression problem, to find x and y from input signals
# y_cla is for the classification problem, to find location_coded from input signals
X = df_cleaned.drop(labels={'x', 'y', 'location_coded'}, axis=1)
y_reg = df_cleaned[['x', 'y']]
y_cla = df_cleaned[['location_coded']]


# In[7]:


print(y_reg.nunique())
print(y_reg.describe())
print(y_cla.describe())


# In[8]:


y_cla.nunique()


# In[9]:


X.min()


# In[10]:


X.min().plot(kind='bar')
plt.title('Minimum Values of Features')
plt.ylabel('Minimum Value')
plt.xticks(fontsize=5)
plt.show()


# In[11]:


X.max().plot(kind='bar')
plt.title('Maximum Values of Features')
plt.ylabel('Maximum Value')
plt.xticks(fontsize=5)
plt.show()


# In[12]:


max_index = X.max().idxmax()
print(max_index)
print(X[max_index].max())


# There was one feature with a positive 'max' value. Is that an outlier? Is that a wrong measurement? Let's see if there are other features with positive max values.

# In[13]:


maxes = X.max()
max_positive_indexes = maxes[maxes > 0].index
print(max_positive_indexes)


# It was only that one feature. Let's look at all the values for that feature.

# In[14]:


sus_feature = X[max_index] # WifiAccessPoint_29
#sus_feature.plot(kind='bar')
#plt.title('Values of the sus feature (WifiAccessPoint_29)')
#plt.ylabel('Value')
#plt.show()
import seaborn as sns
plt.xticks(rotation=90, fontsize=6)
sns.countplot(x=sus_feature)
plt.show()


# Apparently, there are a few positive values. Let's look at another feature and also look at the total number of positive entries in the dataset.

# In[15]:


some_feature = X['WifiAccessPoint_45']
plt.xticks(rotation=90, fontsize=6)
sns.countplot(x=some_feature)
plt.show()


# In[16]:


some_feature = X['WifiAccessPoint_0']
plt.xticks(rotation=90, fontsize=6)
sns.countplot(x=some_feature)
plt.show()


# In[17]:


X.head()


# In[18]:


print(dir())
# print(globals())


# In[19]:


from mpl_toolkits.mplot3d import Axes3D

# Create a figure and axis
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Get the number of features
num_features = X.shape[1]

# Plot each feature
for i in range(num_features):
    feature = X.iloc[:, i]
    xs = range(len(feature))
    ys = [i] * len(feature)
    zs = feature.values
    ax.plot(xs, ys, zs)

# Set axis labels
ax.set_xlabel('Sample')
ax.set_ylabel('Feature')
ax.set_zlabel('Value')

# Set axis limits
ax.set_xlim([0, X.shape[0]])
ax.set_ylim([0, X.shape[1]])
ax.set_zlim([X.min().min(), X.max().max()])

# Adjust position of z-axis label
ax.zaxis.set_label_coords(0.5, -0.1)

# Set view angle
ax.view_init(elev=30, azim=-60)

# Show the plot
plt.show()


# In[134]:


# Plot ALL data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

X_all = df_cleaned.drop(labels={'location_coded'}, axis=1)

# Create a figure and axis
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Define the update function
def update(azim):
    # Clear the previous plot
    ax.clear()

    # Get the number of features
    num_features = X_all.shape[1]

    # Plot each feature
    for i in range(num_features):
        feature = X_all.iloc[:, i]
        xs = range(len(feature))
        ys = [i] * len(feature)
        zs = feature.values
        ax.plot(xs, ys, zs)

    # Set axis labels
    ax.set_xlabel('Sample')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Value')

    # Set axis limits
    ax.set_xlim([0, X_all.shape[0]])
    ax.set_ylim([0, X_all.shape[1]])
    ax.set_zlim([X_all.min().min(), X_all.max().max()])

    # Adjust position of z-axis label
#     ax.zaxis.set_label_coords(0.5, -0.1)

    # Set the view angle
    ax.view_init(elev=30, azim=azim)

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, 360, 10))

ani.save('all_data.gif', writer='imagemagick')

# Show the plot
plt.show()


# In[130]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

# Define the update function
def update(azim):
    # Clear the previous plot
    ax.clear()

    # Get the number of features
    num_features = X.shape[1]

    # Plot each feature
    for i in range(num_features):
        feature = X.iloc[:, i]
        xs = range(len(feature))
        ys = [i] * len(feature)
        zs = feature.values
        ax.plot(xs, ys, zs)

    # Set axis labels
    ax.set_xlabel('Sample')
    ax.set_ylabel('Feature')
    ax.set_zlabel('Value')

    # Set axis limits
    ax.set_xlim([0, X.shape[0]])
    ax.set_ylim([0, X.shape[1]])
    ax.set_zlim([X.min().min(), X.max().max()])

    # Adjust position of z-axis label
    ax.zaxis.set_label_coords(0.5, -0.1)

    # Set the view angle
    ax.view_init(elev=30, azim=azim)

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, 360, 10))

ani.save('animation.gif', writer='imagemagick')

# Show the plot
plt.show()


# In[20]:


print("Total number of positive entries =", X[X>0].sum().sum())
print("Total number of entries (non-NaN) =", X.count().sum())
print("Total number of entries (NaN) =", X.isnull().sum().sum())
print("Total number of entries (NaN + non-NaN) =", X.count().sum(), " + ", X.isnull().sum().sum(), " =", X.count().sum() + X.isnull().sum().sum())
print("Sanity check: 3830 x 91 =", 3830*91)


# It seems that 19 out of 83047 entries are positive. I don't think this is a mistake. So, no feature removal here.

# In[21]:


X.mean().plot(kind='bar')
plt.title('Mean Values of Features')
plt.ylabel('Mean Value')
plt.show()


# In[22]:


# minimum of all values
X.min().min()


# Let's look at the correlation between features and labels (both location_coded and x,y).

# In[23]:


## Correlation analysis for classification
## X, y_cla
# Calculate correlations using one-hot encoding for the categorical
data = df_cleaned

corr_spearman = data.corr(method='spearman')
sns.heatmap(corr_spearman, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap")
plt.show()

data = pd.get_dummies(data, columns=['location_coded'])
# data = df_cleaned.drop(df_cleaned.columns[30:91], axis=1)
corr_pearson = data.corr(method='pearson')
corr_spearman = data.corr(method='spearman')
corr_kendall = data.corr(method='kendall')

# Draw heatmaps
sns.heatmap(corr_pearson, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Pearson Correlation Heatmap")
plt.show()

sns.heatmap(corr_kendall, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Kendall Correlation Heatmap")
plt.show()

sns.heatmap(corr_spearman, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap")
fig1 = plt.gcf()
plt.show()
# fig1.savefig('hq_heatmap_features_x_y_location_coded_after_onehot.png', dpi=1000)


# In[24]:


data.shape


# ## Relation between location_coded and (x, y)

# Now let's visualize the relation between x, y; and location_coded. To do this, I need to convert the categorial values to numerics.

# In[28]:


locations = df_cleaned['location_coded']
plt.xticks(rotation=90, fontsize=4)
sns.countplot(x=locations)
plt.show()

x_values = df_cleaned['x']
sns.histplot(x=x_values)
plt.show()

y_values = df_cleaned['y']
sns.histplot(x=y_values)
plt.show()


# In[30]:


# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Plot x vs y on the bottom left subplot
axs[1, 0].scatter(x_values, y_values, alpha=0.5)
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')

# Plot the histogram of x on the top left subplot
axs[0, 0].hist(x_values, bins=50, alpha=0.5, color='red')
axs[0, 0].set_ylabel('Frequency')

# Plot the histogram of y on the bottom right subplot
axs[1, 1].hist(y_values, bins=50, alpha=0.5, color='green')
axs[1, 1].set_xlabel('Frequency')

# Remove the top right subplot
fig.delaxes(axs[0, 1])

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Show the plot
plt.show()


# In[26]:


# Goal: group location_coded values into 92 groups,
# then for each group get four values: x_min, x_max, y_min, y_max,
# then draw 92 rectangles
# data is in df_cleaned

# group by target and aggregate x and y columns
grouped = df_cleaned.groupby('location_coded').agg({'x': ['min', 'max'], 'y': ['min', 'max']})
print(grouped)
# flatten the column names
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
print(grouped)
# reset the index
grouped = grouped.reset_index()

# print the resulting dataframe
print(grouped)

# create a scatter plot of the data
plt.scatter(df_cleaned['x'], df_cleaned['y'])

from matplotlib.patches import Rectangle

# loop through each group and add a rectangle to the plot
for i, row in grouped.iterrows():
    x_min, x_max = row['x_min'], row['x_max']
    y_min, y_max = row['y_min'], row['y_max']
    rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False)
    plt.gca().add_patch(rect)

# display the plot
plt.show()


# In[32]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_cla_num = le.fit_transform(y_cla['location_coded'])

# Create a larger figure
fig = plt.figure(figsize=(15, 15))

# to check to see if LabelEncoder has reduced the number of unique categories --> it hasn't! :)
print(len(np.unique(y_cla_num)))
plt.scatter(y_reg['x'], y_reg['y'], c=y_cla_num, cmap='jet')
plt.xlim(0,30)
plt.ylim(0,100)
plt.xticks(np.arange(0,31,5))
plt.yticks(np.arange(0,101,5))
plt.colorbar().set_label('location_coded')
plt.xlabel('x')
plt.ylabel('y')

# loop through each group and add a rectangle to the plot
for i, row in grouped.iterrows():
    x_min, x_max = row['x_min'], row['x_max']
    y_min, y_max = row['y_min'], row['y_max']
    rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, fill=False)
    plt.gca().add_patch(rect)

# Set the aspect ratio to be equal
plt.gca().set_aspect('equal', adjustable='box')
# plt.gca().set_aspect('equal')

# Rotate the plot by 90 degrees
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.gca().set_ylim([max(y_reg['y']), min(y_reg['x'])])

# Show the plot
plt.show()


# Let's normalize X and take care of NaNs

# ## Data Imputation
# Here we try 4 different types of data imputation:
# 1) Replace NaNs with minimum of all measurements.
# 2) Repalce NaNs with a value much lower than the minimum.
# 3) Replace NaNs with the overall mean.
# 4) Replace NaNs of each feature, with mean of that feature.

# In[34]:


from sklearn.impute import SimpleImputer

# fill NaN with min value
X_impute_to_min = X.fillna(X.min().min())
X_impute_to_minus_inf = X.fillna(-1000)
X_impute_to_overall_mean = X.fillna(X.mean().mean())
X_impute_to_feature_mean = X.fillna(value=X.mean())

# another way to do it, the result will be equal to X_impute_to_min
imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value=X.min().min())
imputer.fit(X)
X_impute_to_min_test = pd.DataFrame(imputer.transform(X), columns=X.columns)


# # Model Selection
# ### Classification
# Trying different models for classification.

# In[35]:


# X, y_cla
# not normalized
# cross validation
# Standard scaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

scaler =  StandardScaler()

# ML algorithms 
logistic_regression = LogisticRegression(solver='lbfgs')
support_vector_machine = SVC() 
random_forest = RandomForestClassifier(n_estimators=100)
adaboost50 = AdaBoostClassifier(n_estimators=50)
adaboost100 = AdaBoostClassifier(n_estimators=100)
gbc = GradientBoostingClassifier()
knn1 = KNeighborsClassifier(n_neighbors=1)
knn2 = KNeighborsClassifier(n_neighbors=2)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn10 = KNeighborsClassifier(n_neighbors=10)
dtc = DecisionTreeClassifier(random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
lgbm = LGBMClassifier(objective='multiclass', random_state=42)

models = [('Logistic Regression',logistic_regression),
          ('Support Vector Machine', support_vector_machine),
          ('Random Forest',random_forest),
          ('AdaBoost (n=50)', adaboost50),
          ('AdaBoost (n=100)', adaboost100),
#           ('Gradient Boosting Classifier', gbc),
          ('KNN (k=1)', knn1),
          ('KNN (k=2)', knn2),
          ('KNN (k=5)', knn5),
          ('KNN (k=10)', knn10),
          ('Decision Tree Classifier', dtc),
          ('Multilayer Perceptron', mlp),
          ('LightGBM', lgbm)]


# In[79]:


# Temporarily suppress warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

start_time = time.time()

# Let's look at the classifiers and their performance
results = pd.DataFrame(columns=['Classifier', 'Accuracy (mean)', 'Accuracy (std)'])
print("Different models; 5-fold CV; missing -> min; scaler = standard; features = 91 (all)")
for clf_name,clf in models:
    pipe = Pipeline([('transformer', scaler), ('estimator', clf)])
    scores = cross_val_score(pipe, X_impute_to_min, y_cla, cv = 5, scoring = "accuracy")
    new_row = {'Classifier': clf_name, 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores),2)}
    results = results.append(new_row, ignore_index=True)
#     print(scores)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

results.head(100)


# In[80]:


tt = results.sort_values('Accuracy (mean)', ascending=False).head(20)
print(tt)


# In[28]:


# Let's compare standard vs minmaxscaler
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler()

start_time = time.time()

results = pd.DataFrame(columns=['Classifier', 'Accuracy (mean)', 'Accuracy (std)'])
print("Different models; 5-fold CV; missing -> min; scaler = MinMax; features = 91 (all)")
for clf_name,clf in models:
    pipe = Pipeline([('transformer', minmax_scaler), ('estimator', clf)])
    scores = cross_val_score(pipe, X_impute_to_min, y_cla, cv = 5, scoring = "accuracy")
    new_row = {'Classifier': clf_name, 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores),2)}
    results = results.append(new_row, ignore_index=True)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

results.head(10)


# In[73]:


# Let's try out different ways of data imputation for random forest
datas = [('Min (-145)', X_impute_to_min),
         ('-1000', X_impute_to_minus_inf),
         ('Mean (feature)', X_impute_to_feature_mean),
         ('Mean (overall)', X_impute_to_overall_mean)
        ]

results = pd.DataFrame(columns=['Imputation', 'Accuracy (mean)', 'Accuracy (std)'])
print("Different imputations; Model = RF; 5-fold CV; scaler = standard; features = 91 (all)")
for X_name,X in datas:
    pipe = Pipeline([('transformer', scaler), ('estimator', knn1)])
    scores = cross_val_score(pipe, X, y_cla, cv = 5, scoring = "accuracy")
    new_row = {'Imputation': X_name, 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores),2)}
    results = results.append(new_row, ignore_index=True)
results.head(10)


# Conclusions:
# - Model: Random Forest and Logistic Regression are the best.
# - Imputation: replacing NaNs with minimum value or lower is the best.
# 
# Now that random forest is the best model, let's look at feature importance to see which features are the most important ones according to the random forest model.

# In[30]:


from sklearn.model_selection import KFold

## feature importance using random forest and 5-fold cross validation
# using already built random_forest

# Create a K-Fold cross-validator with 5 folds
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the feature importance scores
feature_importance_scores = []

# Loop over the cross-validation folds
for train_index, test_index in cv.split(X):

    # Get the training and test sets for this fold
    X_train, X_test = X_impute_to_min.iloc[train_index], X_impute_to_min.iloc[test_index]
    y_train, y_test = y_cla.iloc[train_index], y_cla.iloc[test_index]

    # Fit the model on the training set for this fold
    random_forest.fit(X_train, y_train)

    # Get the feature importance scores for this fold
    feature_importance_fold = random_forest.feature_importances_

    # Append the scores to the list of feature importance scores
    feature_importance_scores.append(feature_importance_fold)

# Calculate the mean and standard deviation of the feature importance scores
feature_importance_mean = np.mean(feature_importance_scores, axis=0)
feature_importance_std = np.std(feature_importance_scores, axis=0)

# Sort the feature importance scores in descending order
sorted_idx = np.argsort(feature_importance_mean)[::-1]

# Print the feature importance scores in descending order, along with their standard deviation
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {feature_importance_mean[idx]:.3f} +/- {feature_importance_std[idx]:.3f}")


# In[81]:


# Let's do recursive feature elimination (RFE) with random forest
from sklearn.feature_selection import RFE

# initializing X and y
X = X_impute_to_min
y = y_cla

# avoid going crazy locally
import socket
hostname = socket.gethostname()
if hostname == 'ads12team3b-0':
    n_features_start = 1    
else:
    n_features_start = 1

start_time = time.time()

# Create a Recursive Feature Elimination object with a range of n_features_to_select values
n_features_range = np.arange(n_features_start, X.shape[1]+1, 20)
accuracies = []
for n_features in n_features_range:
    rfe = RFE(estimator=random_forest, n_features_to_select=n_features, step=1)

    # Fit the RFE object on the data and use it to transform the data
    rfe.fit(X, y)
    X_transformed = rfe.transform(X)

    # Perform 5-fold cross-validation on the transformed data and record the mean accuracy
    cv_scores = cross_val_score(random_forest, X_transformed, y, cv=5)
    cv_accuracy = np.mean(cv_scores)
    
    print(f"n = {n_features}, accuracy = {cv_accuracy:.2f}")
    accuracies.append(cv_accuracy)


end_time = time.time()

# Print the total execution time
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Plot the accuracy against the number of selected features
plt.plot(n_features_range, accuracies)
plt.title("Accuracy vs. Number of Selected Features")
plt.xlabel("Number of Selected Features")
plt.ylabel("Accuracy")
plt.show()


# Let's change the scaler and see if that would help.
# We keep the best model (i.e., random forest) and imputation to minimum, but only try different scalers from sklearn.

# In[74]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer

minmax_scaler = MinMaxScaler()
maxabs_scaler = MaxAbsScaler()
robust_scaler = RobustScaler()
fx_scaler = FunctionTransformer(func=lambda x: x)  # Identity function

scalers = [('StandardScaler', scaler),
          ('MinMaxScaler', minmax_scaler),
          ('MaxAbsScaler', maxabs_scaler),
          ('RobustScaler', robust_scaler),
          ('NoScaler', fx_scaler)]

print("Different scalers; 5-fold CV; model = RF; missing -> min; features = 91 (all)")
results = pd.DataFrame(columns=['Scaler', 'Accuracy (mean)', 'Accuracy (std)'])
for sc_name,sc in scalers:
    pipe = Pipeline([('transformer', sc), ('estimator', knn1)])
    scores = cross_val_score(pipe, X_impute_to_min, y_cla, cv = 5, scoring = "accuracy")
    new_row = {'Scaler': sc_name, 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores),2)}
    results = results.append(new_row, ignore_index=True)
results.head(10)


# In[33]:


# get classification report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X_impute_to_min, y_cla, test_size=0.2, random_state=42)
random_forest2 = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
y_pred = random_forest2.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[34]:


report = classification_report(y_test, y_pred, output_dict=True)

# Get the macro-averaged precision, recall, and F1-score
macro_precision = report['macro avg']['precision']
macro_recall = report['macro avg']['recall']
macro_f1_score = report['macro avg']['f1-score']

# Print the macro-averaged precision, recall, and F1-score
print("Macro-averaged Precision: {:.2f}".format(macro_precision))
print("Macro-averaged Recall: {:.2f}".format(macro_recall))
print("Macro-averaged F1-score: {:.2f}".format(macro_f1_score))


# In[85]:


# Feature elimination for Random Forest
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

start_time = time.time()

# Define the feature elimination methods to compare
methods = [
#     ('RFE', RFE(random_forest, n_features_to_select=2)),
#     ('PCA(n=2)', PCA(n_components=2)),
#     ('PCA(n=3)', PCA(n_components=5)),
#     ('PCA(n=4)', PCA(n_components=5)),
#     ('PCA(n=10)', PCA(n_components=10)),
#     ('PCA(n=20)', PCA(n_components=20)),
#     ('PCA(n=30)', PCA(n_components=30)),
#     ('PCA', PCA()),
#     ('SelectKBest(k=2)', SelectKBest(mutual_info_classif, k=2)),
#     ('SelectKBest(k=5)', SelectKBest(mutual_info_classif, k=5)),
#     ('SelectKBest(k=10)', SelectKBest(mutual_info_classif, k=10)),
    ('SelectKBest(k=20)', SelectKBest(mutual_info_classif, k=20)),
    ('SelectKBest(k=25)', SelectKBest(mutual_info_classif, k=25)),
    ('SelectKBest(k=30)', SelectKBest(mutual_info_classif, k=30)),
]

# Compare the performance of each method
print("Feature extraction/elimination; 5-fold CV; model = RF; missing -> min; scaler = standard;")
results = pd.DataFrame(columns=['Feature Selection', 'Accuracy (mean)', 'Accuracy (std)'])
for name, method in methods:
    pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selection', method),
        ('classifier', knn1)
    ])
    scores = cross_val_score(pipeline, X_impute_to_min, y_cla, cv=5, scoring='accuracy')
    new_row = {'Feature Selection': name, 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores),2)}
    results = results.append(new_row, ignore_index=True)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
results.head(10)


# In[113]:


# let's look at PCA component values
n = 2
pca = PCA(n_components=n)
x_pca = pca.fit_transform(X_impute_to_min)
for i in range(0, n):    
    plt.plot(pca.components_[i], label=f'PCA comp{i+1}')
plt.title('PCA components')
plt.ylabel('feature coefficients in each component (-1, 1)')
plt.xlabel('features (91)')
plt.legend(loc='upper right')
plt.show()


# In[110]:


sum = 0
for i in range(0,n):
    sum = sum + abs(pca.components_[i])

# get names of columns as vector
feature_names = np.array(X_impute_to_min.columns)

# soft column names by values in sum
features_names_sorted_by_sum_pca = feature_names[np.argsort(sum)[::-1]]

# print sorted column names
print(sum)
print(features_names_sorted_by_sum_pca)


# ## Hyperparameter tuning

# In[84]:


# Hyperparameter tuning using grid search: random forest, logistic regression, KNN
from sklearn.model_selection import GridSearchCV

start_time = time.time()

# define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier())
])

# define the parameter grid
param_grid = [
#     {
#         'pca__n_components': [2, 5, 10, 20, 30],
#         'classifier': [RandomForestClassifier()],
#         'classifier__n_estimators': [100, 200],
#         'classifier__max_depth': [None, 10, 20, 30],
#         'classifier__max_features': ['sqrt', 'log2']
#     },
#     {
#         'pca__n_components': [2, 5, 10, 20, 30],
#         'classifier': [LogisticRegression()],
#         'classifier__penalty': ['l1', 'l2'],
#         'classifier__C': [0.1, 1, 10]
#     },
#     {
#         'pca__n_components': [2, 5, 10, 20, 30],
#         'classifier': [KNeighborsClassifier()],
#         'classifier__n_neighbors': [5, 10, 20],
#         'classifier__weights': ['uniform', 'distance'],
#         'classifier__algorithm': ['ball_tree', 'kd_tree', 'brute']
#     }
    {
        'pca__n_components': [None, 2, 5, 30],
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [1, 2, 5],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
]

# perform the grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# print the best parameters and accuracy score
print(grid_search.best_params_)
print(grid_search.best_score_)


# In[38]:


# access the cv_results_ dictionary
cv_results = grid_search.cv_results_

# display all the results
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"{mean_score:.3f} for {params}")


# In[45]:


# create a DataFrame from the cv_results_ dictionary
results_df = pd.DataFrame(grid_search.cv_results_)
# Sort the results by the rank_test_score
results_df = results_df.sort_values(by='rank_test_score')

# Select the columns you want to keep in the final dataframe
# columns_to_keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']

# Create the final dataframe with the selected columns
# results_df = results_df[columns_to_keep]

# display the results in a pretty table
# print(results_df.to_html())
# print(results_df.to_markdown())
results_df.head(20)


# In[46]:


results_df.tail()


# ## Outlier detection (DBSCAN)

# In[86]:


# Outlier detection
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

X = X_impute_to_min
y = y_cla

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 5 components
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)
# Create an instance of DBSCAN with parameters eps and min_samples
dbscan = DBSCAN(eps=0.5, min_samples=5)

# Fit the DBSCAN model to the data
dbscan.fit(X_pca)

# Get the predicted labels (-1 indicates noise points)
labels = dbscan.labels_

# Print the number of clusters and number of noise points
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Visualize the clusters and noise points
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[87]:


# Remove the noise points from the data
X_v = X.values
X_no_noise = X_v[labels != -1, :]
y_v = y_cla.values
y_no_noise = y_v[labels != -1, :]

# Print the number of clusters and number of noise points
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('Number of points after removing noise: %d' % X_no_noise.shape[0])

# Visualize the clusters and noise points
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[88]:


print(type(X_no_noise))
print(X_no_noise.shape)


# In[89]:


# let's do random forest with no noise
X = X_no_noise
y = y_no_noise
print(X.shape)
print(y.shape)
pipe = Pipeline([('transformer', scaler), ('estimator', random_forest)])
scores = cross_val_score(pipe, X, y, cv = 5, scoring = "accuracy")
print(round(np.mean(scores),2))
print(round(np.std(scores),3))


# In[92]:


# Let's do KNN (k=1) with no noise
pipe = Pipeline([('transformer', scaler), ('estimator', knn1)])
scores = cross_val_score(pipe, X, y, cv = 5, scoring = "accuracy")
print(round(np.mean(scores),2))
print(round(np.std(scores),2))


# ## Remove the first feature and redo classification

# In[37]:


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

X_impute_to_min_dropped_first = X_impute_to_min.drop(labels={'WifiAccessPoint_0'}, axis=1)

start_time = time.time()

# Let's look at the classifiers and their performance
results = pd.DataFrame(columns=['WifiAccessPoint_0', 'Accuracy (mean)', 'Accuracy (std)'])
print("with/without 1st feature; random forest; 5-fold CV; missing -> min; scaler = standard")
pipe = Pipeline([('transformer', scaler), ('estimator', random_forest)])

# with 1st feature
scores = cross_val_score(pipe, X_impute_to_min, y_cla, cv = 5, scoring = "accuracy")
new_row = {'WifiAccessPoint_0': 'with', 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores)**2,3)}
results = results.append(new_row, ignore_index=True)

# without 1st feature
scores = cross_val_score(pipe, X_impute_to_min_dropped_first, y_cla, cv = 5, scoring = "accuracy")
new_row = {'WifiAccessPoint_0': 'without', 'Accuracy (mean)': round(np.mean(scores),2), 'Accuracy (std)': round(np.std(scores)**2,3)}
results = results.append(new_row, ignore_index=True)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

results.head()


# ## Regression

# In[140]:


# Set X and y
X = X_impute_to_min
y = y_reg

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ML algorithms 
sgd_regression = SGDRegressor()
support_vector_machine = SVR() 
random_forest = RandomForestRegressor(n_estimators=100)
adaboost50 = AdaBoostRegressor(n_estimators=50)
adaboost100 = AdaBoostRegressor(n_estimators=100)
knn1 = KNeighborsRegressor(n_neighbors=1)
knn2 = KNeighborsRegressor(n_neighbors=2)
knn3 = KNeighborsRegressor(n_neighbors=3)
knn4 = KNeighborsRegressor(n_neighbors=4)
knn5 = KNeighborsRegressor(n_neighbors=5)
knn6 = KNeighborsRegressor(n_neighbors=6)
knn10 = KNeighborsRegressor(n_neighbors=10)
dtr = DecisionTreeRegressor(random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
lgbm = LGBMRegressor(random_state=42)

multi_output_models = [
          ('Random Forest',random_forest),
          ('KNN (k=1)', knn1),
          ('KNN (k=2)', knn2),
          ('KNN (k=3)', knn3),
          ('KNN (k=4)', knn4),
          ('KNN (k=5)', knn5),
          ('KNN (k=6)', knn6),
          ('KNN (k=10)', knn10),
          ('Decision Tree Regressor', dtr),
          ('Multilayer Perceptron', mlp),
 ]

single_output_models = [
          ('Support Vector Machine', support_vector_machine),
          ('SGD Regressor',sgd_regression),
          ('AdaBoost (n=50)', adaboost50),
          ('AdaBoost (n=100)', adaboost100),
          ('LightGBM', lgbm)    
]


# In[67]:


# Temporarily suppress warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

from scipy.spatial import distance

# Calculate Euclidean distance
def mean_euclidean_distance(y_true, y_pred):
    distances = [distance.euclidean(y_true[i], y_pred[i]) for i in range(len(y_true))]
    print(distances)
    return sum(distances) / len(distances)

# Define new scorer based on Euclidean distance
euclidean_scorer = make_scorer(euclidean_distance, greater_is_better=False)

start_time = time.time()

scaler =  StandardScaler()

print("Different models; 5-fold CV; missing -> min; scaler = standard; features = 91 (all)")

# create an empty results dataframe
results = pd.DataFrame(columns=['Regressor',
                                'RMSE (mean)', 'RMSE (std)',
                                'R2 (mean)', 'R2 (std)',
                                'E-dist (mean)', 'E-dist (std)'])

# loop through the multi-output models
for clf_name, clf in multi_output_models:
    pipe = Pipeline([('transformer', scaler), ('estimator', clf)])
    # metric: negative root mean squared error
    rmse_scores = cross_val_score(pipe, X, y, cv=5, scoring=make_scorer(mean_squared_error))
    mean_rmse = round(np.mean(np.sqrt(rmse_scores)), 2)
    std_rmse = round(np.std(np.sqrt(rmse_scores)), 2)
    # metric: R2
    r2_scores = cross_val_score(pipe, X, y, cv=5, scoring=make_scorer(r2_score))
    mean_r2 = round(np.mean(r2_scores), 2)
    std_r2 = round(np.std(r2_scores), 2)
    # metric: MAE
    mae_scores = cross_val_score(pipe, X, y, cv=5, scoring=make_scorer(mean_absolute_error))
    mean_mae = round(np.mean(mae_scores), 2)
    std_mae = round(np.std(mae_scores), 2)
    # metric: Euclidean distance
    edist_scores = cross_val_score(pipe, X, y, cv=5, scoring=euclidean_scorer)
    print(edist_scores)
    mean_edist = round(np.mean(edist_scores), 2)
    std_edist = round(np.std(edist_scores), 2)
    # create a new row for the results dataframe
    new_row = {
        'Regressor': clf_name,
        'RMSE (mean)': mean_rmse, 'RMSE (std)': std_rmse,
        'R2 (mean)': mean_r2, 'R2 (std)': std_r2,
        'MAE (mean)': mean_mae, 'MAE (std)': std_mae,
        'E-dist (mean)': mean_edist, 'E-dist (std)': std_edist
    }
    # add the new row to the results dataframe, or update an existing row if the regressor already exists
    if clf_name in results['Regressor'].values:
        idx = results.index[results['Regressor'] == clf_name].tolist()[0]
        results.loc[idx] = new_row
    else:
        results = results.append(new_row, ignore_index=True)

# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")

results.sort_values('E-dist (mean)').head(100)
# results.head(100)

# print the results dataframe
# print(results)


# # Dependency to unseen data

# In[138]:


def compare_knn_and_rf(X_train, X_test, y_train, y_test):
    print(X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]))

    # Plot the data
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(y_train['x'], y_train['y'], c='blue', label='Training Set')
    plt.scatter(y_test['x'], y_test['y'], c='red', label='Testing Set')
    plt.xlim(0,30)
    plt.ylim(0,100)
    plt.xticks(np.arange(0,31,5))
    plt.yticks(np.arange(0,101,5))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')

    ## Step 2: try KNN and random forest on the split
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RF MAE: {mae:.2f}")
    r2 = r2_score(y_test, y_pred)
    print(f"RF R2: {r2:.2f}")

    plt.scatter(y_pred[:,0], y_pred[:,1], c='green', marker='*', label='Predicted by Random Forest')

    knn1 = KNeighborsRegressor(n_neighbors=1)
    knn1.fit(X_train, y_train)
    y_pred = knn1.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"KNN(n=1) MAE: {mae:.2f}")
    r2 = r2_score(y_test, y_pred)
    print(f"KNN(n=1) R2: {r2:.2f}")

    plt.scatter(y_pred[:,0], y_pred[:,1], c='yellow', marker='+', label='Predicted by KNN')
    plt.legend()
    plt.show()


# In[139]:


# Regression data: X, y
# Models to compare: knn1, random_forest
from sklearn.model_selection import train_test_split

## Split 1: random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
compare_knn_and_rf(X_train, X_test, y_train, y_test)

## Split 2: non-random split (creates unseen data)
# train_size = 0.9
# train_idx = int(train_size * X.shape[0])
# X_train, X_test = X[:train_idx], X[train_idx:]
# y_train, y_test = y[:train_idx], y[train_idx:]
# compare_knn_and_rf(X_train, X_test, y_train, y_test)


# In[129]:


## Split 3: non-random split (creates unseen data)
# Create a mask to select rows with y less than 100
mask = y['y'] < 90
X_train = X[mask]
y_train = y[mask]
X_test = X[~mask]
y_test = y[~mask]
compare_knn_and_rf(X_train, X_test, y_train, y_test)


# In[130]:


## Split 4: non-random split (creates unseen data)
# Create a mask to select rows with y less than 100
mask = y['x'] < 26
X_train = X[mask]
y_train = y[mask]
X_test = X[~mask]
y_test = y[~mask]
compare_knn_and_rf(X_train, X_test, y_train, y_test)


# In[142]:


## Split 5: non-random split (creates unseen data)
# Create a mask to select rows with y less than 100
mask = (y['x'] < 20) | (y['x'] > 25) | (y['y'] < 80) | (y['y'] > 90)
X_train = X[mask]
y_train = y[mask]
X_test = X[~mask]
y_test = y[~mask]
compare_knn_and_rf(X_train, X_test, y_train, y_test)


# In[143]:


start_time = time.time()
for clf_name, clf in multi_output_models:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(clf_name)
    print(f"RF MAE: {mae:.2f}")
    print(f"RF R2: {r2:.2f}")    # create a new row for the results dataframe
    
# Print the total execution time
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")


# In[ ]:




