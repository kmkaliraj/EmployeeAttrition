
# coding: utf-8

# # Machine Learning: HR Data Analysis - Predict Attrition of an Employee

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# ## Load data

# In[2]:

employees = pd.read_csv('hr_employee_dataset.csv')
employees.head()
employees = employees.rename(columns = {'sales': 'department'})
employees.shape


# ## Normalize Data

# In[3]:

employees['salary'].unique()
employees['salary'] = pd.factorize(employees['salary'])[0]
employees['department'].unique()
employees['department'] = pd.factorize(employees['department'])[0]
employees.head()


# ## Correlation Matrix for finding relevant features

# In[11]:

correlation_matrix = employees.corr()
plt.subplots(figsize = (8, 8))

sns.heatmap(correlation_matrix, vmax = .8, square = True)
plt.show()


# ## Extract prediction and predictive features

# In[4]:

leave_result = employees['left']
y = np.where(leave_result == 1, 1, 0)
X = employees.drop('left', axis = 1).as_matrix().astype(np.float)


# ## Scaling Features

# In[10]:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels: ", np.unique(y))


# ## Applying Different Classification Models

# In[6]:

from sklearn.model_selection import KFold
import time
from sklearn.metrics import roc_curve, auc
from scipy import interp

# Results: {'pred': y_pred, 'time': seconds, true-positive-rate': tpr, 'false-positive-rate': fpr }

def run_clf(X, y, clf):
    start = time.time()
    
    y_pred = y.copy()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    t = 0.0
    
    # Construct a kfolds object
    kf = KFold(n_splits = 3, shuffle = True)
    
    # Iterate through folds
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        
        clf.fit(X_train, y_train)
        
        y_pred[test_index] = clf.predict(X_test)
        fpr, tpr, _ = roc_curve(y[test_index], y_pred[test_index])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
    
    mean_tpr /= kf.get_n_splits(X)
    mean_tpr[-1] = 1.0
        
    end = time.time()
    t = end - start
    
    return { 'y_pred': y_pred, 
             'time': t, 
             'true-positive-rate': mean_tpr, 
             'false-positive-rate': mean_fpr }


# In[24]:

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import average_precision_score

classifiers = {
    "Logistic Regression":LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors = 3),
    "Linear SVM": SVC(kernel = "linear", C = 0.025),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 18),
    "Neural Network": MLPClassifier(alpha = 1),
}


def runall():
    results = {}
    for clf_name, clf in classifiers.items():
        results[clf_name] = run_clf(X, y, clf)
        print("{0:30}: {1:>7.3f}s".format(clf_name, results[clf_name]['time']))
    return results
        
results = runall()


# ## Different models and their accuracy

# In[14]:

def accuracy(y_true, y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)

for clf_name, result in results.items():
    print("{0:30}: {1:.3f}".format(clf_name, accuracy(y, result['y_pred'])))


# ## Confusion Matrix

# In[15]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def draw_confusion_matrices(cms, classes):
    fig = plt.figure(figsize = (10, 15))
    
    i = 1   # used to compute the matrix location
    for clf_name, cm in cms.items():
        thresh = cm.max() / 2   # used for the text color
        
        ax = fig.add_subplot(len(cms) / 2 + 1, 2, i,
                             title = 'Confusion matrix for %s' % clf_name, 
                             xlabel = 'Predicted',
                             ylabel = 'True')
        cax = ax.matshow(cm, cmap = plt.cm.Blues)
        fig.colorbar(cax)
        i += 1
        
        # Ticks
        ax.set_xticklabels([''] + classes)
        ax.set_yticklabels([''] + classes)
        ax.tick_params(labelbottom = True, labelleft = True, labeltop = False)
        
        # Text
        for x in range(len(cm)):
            for y in range(len(cm[0])):
                ax.text(y, x, cm[x, y], 
                        horizontalalignment = 'center', 
                        color = 'black' if cm[x, y] < thresh else 'white')
        
    plt.tight_layout()
    plt.show()

matrices = {}
for clf_name, result in results.items():
    matrices[clf_name] = confusion_matrix(y, result['y_pred'])

labels = np.unique(y).tolist()
draw_confusion_matrices(matrices, labels)


# In[ ]:



