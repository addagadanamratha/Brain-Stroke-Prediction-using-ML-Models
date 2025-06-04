#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[79]:


import pandas as pd #for dataframe
import numpy as np #for numerical calculations
import seaborn as sns #for visualisation purposes
import matplotlib.pyplot as plt #is a 2D plotting library for Python.
import warnings #allows you to control the behavior of warnings in your code.
from datetime import datetime #used for manipulating dates and times.

################### Sklearn ####################################
from sklearn.preprocessing import MinMaxScaler #or scaling numerical features to a specified range, typically between 0 and 1.
from sklearn.model_selection import train_test_split, GridSearchCV #is used to split datasets into training and testing sets, 
from sklearn.ensemble import RandomForestClassifier #Random Forest is an ensemble learning method for classification.
from sklearn import metrics #includes functions for evaluating the performance of machine learning models
from sklearn.linear_model import LogisticRegression #which is used for logistic regression.
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier #which is used for decision tree-based classification.
from sklearn.neighbors import KNeighborsClassifier #which is used for k-nearest neighbors classification.


# ### Library configurations

# In[80]:


pd.options.mode.copy_on_write = True # Allow re-write on variable
sns.set_style('darkgrid') # Seaborn style
warnings.filterwarnings('ignore') # Ignore warnings
pd.set_option('display.max_columns', None) # Setting this option will print all collumns of a dataframe
pd.set_option('display.max_colwidth', None) # Setting this option will print all of the data in a feature


# ### Collecting Data

# - At first, import dataset in csv format by pandas library and read_csv method.

# In[81]:


data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data.head()


# ### Data Informations
# - We drop id columns, because its a unique identifier number.

# In[82]:


# Drop column = 'id'
#Here we are as it doesnot contribute meaningful information to the analysis.
#It is redundant and can  be removed from the dataset.
#Reduces the dimentionality of the dataset.
data.drop(columns='id', inplace=True)


# In[83]:


data.head()


# In[84]:


data.info()


# In[85]:


#describe():method computes summary statistics for numerical columns in the DataFrame.
#include='all' parameter ensures that both numerical and categorical columns are included in the summary statistics.
#round(..., 2) is used to round the numerical values to two decimal places for better readability.
round(data.describe(include='all'), 2)


# - We have 5110 samples , with no null values

# ###  Handling Missing Values

# In[86]:


data.isna().sum()


# In[87]:


#calculating the percentage of missing values for each column in the DataFrame and prints the result.
#useful for understanding the proportion of missing values in each column.
print((data.isna().sum()/len(data))*100)


# - There is 201 samples with no values in bmi column , its about 4% of data. For better result we drop them.

# In[88]:


### Missing values in BMI columns is about 4% , we drop them.
data.dropna(how='any', inplace=True)

#how='any' specifies that a row should be dropped if it contains at least one missing value.


# ### Visualization and Plots

# In[89]:


#Separating the independent variable and target variable
#This is often done to prepare the data for training a machine learning model
# The idea is that you want to use the features to train the model to predict the target variable.
cols = data.columns[:-1]
cols


# In[90]:


data


# In[91]:


#contain the names of columns categorized as numerical and categorical, respectively.
numeric_columns = ['age', 'bmi', 'avg_glucose_level']
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']


# In[92]:


#This code different types of plots (KDE plot, boxplot, and scatter plot) for numerical columns in a dataset.

i = 0
fig, ax = plt.subplots(3, 3, figsize=(15, 8)) #Creating a 3x3 grid
plt.subplots_adjust(hspace = 0.5) #Adjusting the vertical space between subplots for better readability.
for num_col in numeric_columns :
    sns.kdeplot(x=num_col, hue='stroke', data=data, multiple='stack', ax=ax[i,0]) #Creating a Kernel Density Estimate (KDE) plot
    sns.boxplot(x=num_col, data=data, ax=ax[i, 1]) #Creating a boxplot
    sns.scatterplot(x=num_col, y='stroke', data=data, ax=ax[i, 2]) #Creating a scatter plot
    i+=1 #Incrementing the counter variable i to move to the next row of subplots.
plt.show()


# In[93]:


#This code generates pairs of count plots for each categorical column in the dataset. 

i=0
while i<8 :
    
    # Left AX
    fig = plt.figure(figsize=(10, 4)) #Creating a new figure for the pair of count plots with a specific size.
    plt.subplot(1, 2, 1) #Creating the left subplot
    plt.title(categorical_columns[i], size=20, weight='bold', color='navy') # Setting the title of the left subplot
    ax = sns.countplot(x=categorical_columns[i], data=data)
    ax.bar_label(ax.containers[0])
    ax.tick_params(axis='x', rotation=300)
    i+=1
    
    # Right AX
    plt.subplot(1, 2, 2) #Creating the right subplot 
    plt.title(categorical_columns[i], size=20, weight='bold', color='navy')
    ax = sns.countplot(x=categorical_columns[i], data=data)
    ax.bar_label(ax.containers[0])
    ax.tick_params(axis='x', rotation=300)
    i+=1
    plt.show()


# In[94]:


x = data['stroke'].value_counts() #counts the occurrences of each unique value in the 'stroke' column of the DataFrame 


explode = [0, 0.15] #second wedge, for STROKE=1 is exploded by 15% of the radius.
labels = ['Stroke=0', 'Stroke=1']
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

plt.pie(x, explode=explode, shadow=True, autopct='%1.1f%%', labels=labels, textprops=dict(color="w", weight='bold', size=15))
plt.legend()
plt.show()


# ## Plots Analysis
# ### Results:
# - About 96% of samples have not Stroke and 4% have stroke.
# - Distribution of samples is a Normal distribution.
# - Those who have had a stroke are in:
#         * age in range 40 to 85
#         * bmi in range 20 to 40
#         * glocuse level in range 50 to 130
# - About 60% of samples are female.
# - About 91% of samples dont have any hypertension.
# - About 95% of samples dont have any heart disease.
# - About 34% of samples never get married.
# - Most of samples worked in private.
# - We dont have any information in smoking field for 1483 of sapmples.

# ### Unique Values
# - We count number of unique values in each categorical column, to change them with integer values. Here we use .unique() command.

# In[95]:


#creates a list, containing the names of categorical columns
columns_temp = ['gender', 'ever_married', 'work_type', 'smoking_status', 'Residence_type']

for col in columns_temp :
    print('column :', col)
    for index, unique in enumerate(data[col].unique()) :
        print(unique, ':', index) #prints each unique value in the column along with its index within the column.
    print('_'*45)


# In[96]:


#each block of code replaces categorical values in specific columns with corresponding numerical values

#for gender
data_2 = data.replace(
    {'gender' : {'Male' : 0, 'Female' : 1, 'Other' : 2}}
)

#for ever_married
data_2 =  data_2.replace(
    {'ever_married' : {'Yes' : 0, 'No' : 1}}
)

#for work_type
data_2 =  data_2.replace(
    {'work_type' : {'Private' : 0, 'Self-employed' : 1, 'Govt_job' : 2, 'children' : 3, 'Never_worked' : 4}}
)

#for smoking_status
data_2 =  data_2.replace(
    {'smoking_status' : {'formerly smoked' : 0, 'never smoked' : 1, 'smokes' : 2, 'Unknown' : 3}}
)

#for Residence_type
data_2 =  data_2.replace(
    {'Residence_type' : {'Urban' : 0, 'Rural' : 1}}
)


# In[97]:


data_2.head()


# ### Normalization

# - Define X & y 

# In[98]:


#The resulting DataFrame contains the features (independent variables) that will be
#used to train a model.

X_temp = data_2.drop(columns='stroke')
y = data_2.stroke #epresents the target variable (dependent variable)


# - To decrease effect of larg values, we use MinMaxScaler to normalize X.

# In[99]:


scaler = MinMaxScaler().fit_transform(X_temp)
X = pd.DataFrame(scaler, columns=X_temp.columns)
X.describe()


# ### Step-4:Modeling

# - Initialization

# In[100]:


# define a function to ploting Confusion matrix
def plot_confusion_matrix(y_test, y_prediction):
    cm = metrics.confusion_matrix(y_test, y_prediction)
    ax = plt.subplot()
    ax = sns.heatmap(cm, annot=True, fmt='', cmap="Greens")
    ax.set_xlabel('Prediced labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Dont Had Stroke', 'Had Stroke'])
    ax.yaxis.set_ticklabels(['Dont Had Stroke', 'Had Stroke']) 
    plt.show()


# In[101]:


# Splite X, y to train & test dataset.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)


# ### RandomForestClassifier

# In[102]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_estimators' : [50, 100, 250, 500],
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'max_features' : ['sqrt', 'log2']
}

rf = RandomForestClassifier(n_jobs=-1)
rf_cv = GridSearchCV(estimator=rf, cv=10, param_grid=parameters).fit(X_train, y_train)

print('Tuned hyper parameters : ', rf_cv.best_params_)
print('accuracy : ', rf_cv.best_score_)


# In[103]:


# calculate time befor training the Random Forest model.
t1 = datetime.now()
# Model: initializes rf with best hyperparameters found during grid search & fits in to training data
rf = RandomForestClassifier(**rf_cv.best_params_).fit(X_train, y_train)
# calculate time after after training the Random Forest model.
t2 = datetime.now()


# In[104]:


#This line uses the trained Random Forest model to make predictions on the test data (X_test).
y_pred_rf = rf.predict(X_test)


#This line calculates and prints the accuracy score of the Random Forest model on the test data.
rf_score = round(rf.score(X_test, y_test), 3)
print('RandomForestClassifier score : ', rf_score)


# In[105]:


#These lines calculate and print the time taken to train the Random Forest model.

delta = t2-t1
delta_rf = round(delta.total_seconds(), 3)
print('RandomForestClassifier takes : ', delta_rf, 'Seconds')


# In[106]:


#This line calls the previously defined function 'plot_confusion_matrix' 
#to visualize the confusion matrix for the Random Forest model.

plot_confusion_matrix(y_test, y_pred_rf)


# In[107]:


cr = metrics.classification_report(y_test, y_pred_rf)
print(cr)


# ### LogisticRegression

# In[108]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000], #Inverse of regularization strength.
    'class_weight' : ['balanced'], #adjusting weights inversely proportional to class frequencies.
    'solver' : ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] #Algorithm to use in the optimization problem.
}

#This line initializes a Logistic Regression model (lr) with default hyperparameters.
lr = LogisticRegression()
lr_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10).fit(X_train, y_train)


#These lines print the best hyperparameters and the corresponding accuracy obtained during the grid search.
print('Tuned hyper parameters : ', lr_cv.best_params_)
print('accuracy : ', lr_cv.best_score_)


# In[109]:


# Calculate time befor training the Logistic Regression model.
t1 = datetime.now()
# Model
lr = LogisticRegression(**lr_cv.best_params_).fit(X_train, y_train)
# Calculate time after training the Logistic Regression model.
t2 = datetime.now()


# In[110]:


#This line uses the trained Logistic Regression model to make predictions on the test data 
y_pred_lr = lr.predict(X_test)


#This line calculates and prints the accuracy score of the Logistic Regression model on the test data.
lr_score = round(lr.score(X_test, y_test), 3)
print('LogisticRegression score : ', lr_score)


# In[111]:


#These lines calculate and print the time taken to train the Logistic Regression model.

delta = t2-t1
delta_lr = round(delta.total_seconds(), 3)
print('LogisticRegression takes : ', delta_lr, 'Seconds')


# In[112]:


#This line calls the previously defined function plot_confusion_matrix 
#to visualize the confusion matrix for the Logistic Regression model.

plot_confusion_matrix(y_test, y_pred_lr)


# In[113]:


#These lines calculate and print the classification report, which includes precision, recall, f1-score, and support for each class in the test set.


cr = metrics.classification_report(y_test, y_pred_lr)
print(cr)


# ### SVC

# In[114]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'C' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
    'gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
}



svc = SVC()
svc_cv = GridSearchCV(estimator=svc, param_grid=parameters, cv=10).fit(X_train, y_train)



print('Tuned hyper parameters : ', svc_cv.best_params_)
print('accuracy : ', svc_cv.best_score_)


# In[115]:


# Calculate time befor run algorithm
t1 = datetime.now()
# Model
svc = SVC(**svc_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm
t2 = datetime.now()


# In[116]:


y_pred_svc = svc.predict(X_test)

svc_score = round(svc.score(X_test, y_test), 3)
print('SVC Score : ', svc_score)


# In[117]:


delta = t2-t1
delta_svc = round(delta.total_seconds(), 3)
print('SVC : ', delta_svc, 'Seconds')


# In[118]:


plot_confusion_matrix(y_test, y_pred_svc)


# In[119]:


cr = metrics.classification_report(y_test, y_pred_svc)
print(cr)


# - DecisionTreeClassifier

# In[120]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'criterion' : ['gini', 'entropy', 'log_loss'],
    'splitter' : ['best', 'random'],
    'max_depth' : list(np.arange(4, 30, 1))
        }



tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(estimator=tree, cv=10, param_grid=parameters).fit(X_train, y_train)



print('Tuned hyper parameters : ', tree_cv.best_params_)
print('accuracy : ', tree_cv.best_score_)


# In[121]:


# Calculate time befor run algorithm :
t1 = datetime.now()
# Model :
tree = DecisionTreeClassifier(**tree_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm :
t2 = datetime.now()


# In[122]:


y_pred_tree = tree.predict(X_test)

tree_score = round(tree.score(X_test, y_test), 3)
print('DecisionTreeClassifier Score : ', tree_score)


# In[123]:


delta = t2-t1
delta_tree = round(delta.total_seconds(), 3)
print('DecisionTreeClassifier takes : ', delta_tree, 'Seconds')


# In[124]:


plot_confusion_matrix(y_test, y_pred_tree)


# In[125]:


cr = metrics.classification_report(y_test, y_pred_tree)
print(cr)


# ### KNeighborsClassifier

# In[126]:


# a dictionary to define parameters to test in algorithm
parameters = {
    'n_neighbors' : list(np.arange(3, 20, 2)),
    'p' : [1, 2, 3, 4]
}

# calculate time to run in second
t1 = datetime.now()

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(estimator=knn, cv=10, param_grid=parameters).fit(X_train, y_train)

t2 = datetime.now()

print('Tuned hyper parameters : ', knn_cv.best_params_)
print('accuracy : ', knn_cv.best_score_)


# In[127]:


# Calculate time befor run algorithm :
t1 = datetime.now()
# Model :
knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X_train, y_train)
# Calculate time after run algorithm :
t2 = datetime.now()


# In[128]:


y_pred_knn = knn_cv.predict(X_test)

knn_score = round(knn.score(X_test, y_test), 3)
print('KNeighborsClassifier Score :', knn_score)


# In[129]:


delta = t2-t1
delta_knn = round(delta.total_seconds(), 3)
print('KNeighborsClassifier takes : ', delta_knn, 'Seconds')


# In[130]:


plot_confusion_matrix(y_test, y_pred_knn)


# In[131]:


cr = metrics.classification_report(y_test, y_pred_knn)
print(cr)


# ## Result

# In[132]:


result = pd.DataFrame({
    'Algorithm' : ['RandomForestClassifier', 'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'KNeighborsClassifier'],
    'Score' : [rf_score, lr_score, svc_score, tree_score, knn_score], #accuracy scores for each algorithm
    'Delta_t' : [delta_rf, delta_lr, delta_svc, delta_tree, delta_knn] #contains the time taken (in seconds) for each algorithm.
})

result


# In[133]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x='Algorithm', y='Score', data=result, ax=ax[0])
ax[0].bar_label(ax[0].containers[0], fmt='%.3f')
ax[0].set_xticklabels(labels=result.Algorithm, rotation=300)

sns.barplot(x='Algorithm', y='Delta_t', data=result, ax=ax[1])
ax[1].bar_label(ax[1].containers[0], fmt='%.3f')
ax[1].set_xticklabels(labels=result.Algorithm, rotation=300)
plt.show()


# ### According to the above plots, best algorithms base on Score are :
# 
# 1. RandomForestClassifier
# 2. SVC
# 3. DecisionTreeClassifier
# 4. KNeighborsClassifier
# 
# ### And best Algorithm base on runtime, are :
# 
# - DecisionTreeClassifie
# - KNeighborsClassifier
# 
# 
# ###  ~ We choose  KNeighborsClassifier 

# ### Final Modeling

# In[134]:


knn = KNeighborsClassifier(**knn_cv.best_params_).fit(X, y)
knn


# In[135]:


# returns the accuracy score of the K-Nearest Neighbors classifier on the entire dataset
# The score method compares the predicted labels with the actual labels (y) and computes the accuracy.
knn.score(X, y)

