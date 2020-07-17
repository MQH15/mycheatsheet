# Machine Learning Snippets

## Partition
```python
from sklearn.model_selection import train_test_split
# partition train-test
train_set, test_set = train_test_split(data, test_size=0.4, random_state=42)
# partition with shuffle
train_set, test_set = train_test_split(data, test_size=0.4, random_state=42, shuffle=True)
# partition with stratify
train_set, test_set = train_test_split(data, test_size=0.4, random_state=42, stratify=data["target"])
# partition function
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)
# label function
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)
```


## GridSearch
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
neigh_reg = GridSearchCV(KNeighborsRegressor(),param_grid={"n_neighbors":np.arange(3,50)},cv = 10,scoring = "mean_squared_error",n_jobs=-1)
# Fit will test all of the combinations
neigh_reg.fit(X_train,y_train)
# Best estimator and best parameters
neigh_reg.best_estimator_
neigh_reg.best_params_
neigh_reg.best_score_
```

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dcstree_clf = GridSearchCV(DecisionTreeClassifier(),param_grid={"max_depth":range(1,20),"min_samples_leaf":range(20,100),"min_samples_split":range(20,50)},cv = 10,scoring = "f1_weighted",n_jobs=-1,return_train_score=True)
# Fit will test all of the combinations
dcstree_clf.fit(X_train,y_train)
# Best estimator and best parameters
dcstree_clf.best_estimator_
dcstree_clf.best_params_
dcstree_clf.best_score_
```

```python
# results of GridSearch
cvres = dcstree_clf.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("F1 score:", mean_score, "-", "Parámetros:", params)
```


## RandomizedSearch
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
rgstree_reg = RandomizedSearchCV(DecisionTreeRegressor(),param_distributions={"max_depth":range(1,20),"min_samples_leaf":range(20,100),"min_samples_split":range(20,50)},cv = 10,scoring = "mean_squared_error",n_jobs=-1)
# Fit will test all of the combinations
rgstree_reg.fit(X_train,y_train)
# Best estimator and best parameters
rgstree_reg.best_estimator_
rgstree_reg.best_params_
rgstree_reg.best_score_
```

```python
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
svc_clf = RandomizedSearchCV(SVC(),param_grid={"kernel":["linear","rbf"],"C":range(1,10000,1000)},cv = 10,scoring = "f1_weighted",n_jobs=-1,return_train_score=True)
# Fit will test all of the combinations
svc_clf.fit(X_train,y_train)
# Best estimator and best parameters
svc_clf.best_estimator_
svc_clf.best_params_
svc_clf.best_score_
```


## Cross Validation

```python
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=42)
cross_val_score(SVR(),X,y,cv=cv,scoring="mean_squared_error").mean()
```

```python
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n = X.shape[0], n_folds=10, shuffle=True, random_state=42)
cross_val_score(SVC(),X,y,cv=cv,scoring="f1_weighted").mean()
```


## Regression Algorithms

### Linear Regression with stastmodel
parameters:
- β0,β1,β2,...,βn
```python
# Load the library
import statsmodels.api as sm
# Create an instance of the model
linearstat_reg = sm.OLS(y_train,X_train).fit()
# results
print(linearstat_reg.summary())
# Do predictions
linearstat_reg.predict(X_test)
```

### Linear Regression with sklearn
parameters:
- β0,β1,β2,...,βn
```python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
linearskl_reg = LinearRegression()
# Fit the data
linearskl_reg.fit(X_train,y_train)
# Results
linearskl_reg.intercept_
linearskl_reg.coef_
# Do predictions
linearskl_reg.predict(X_test)
```

### k Nearest Neighbors
parameters:
- n_neighbors
```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance of the model
knn_reg = KNeighborsRegressor(n_neighbors=10)
# Fit the data
knn_reg.fit(X_train,y_train)
# Do predictions
knn_reg.predict(X_test)
```

### Decision Tree
parameters:
- Max_depth: Number of Splits
- Min_samples_leaf: Minimum number of observations per leaf
```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance of the model
rgstree_reg = DecisionTreeRegressor(max_depth=3)
# Fit the data
rgstree_reg.fit(X_train,y_train)
# Do predictions
rgstree_reg.predict(X_test)
```

### Gradient Boosted Trees
parameters:
- n_estimators: number of trees
- learning_rate: steps levels
- max_depth: tree depth
- min_samples_leaf: Minimum number of observations per leaf
```python
from sklearn.ensemble import GradientBoostingRegressor
# Create an instance of the model
gbm_reg = GradientBoostingRegressor(max_depth=4,n_estimators=100, learning_rate=0.1)
# Fit the data
gbm_reg.fit(X_train,y_train)
# Do predictions
gbm_reg.predict(X_test)
```

### Tree Forest
parameters:
- n_estimators: number of trees
- max_depth: tree depth
- min_samples_leaf: Minimum number of observations per leaf
```python
from sklearn.ensemble import RandomForestRegressor
# Create an instance of the model
rndf_reg = RandomForestRegressor(max_depth=4,n_estimators=100)
# Fit the data
rndf_reg.fit(X_train,y_train)
# Do predictions
rndf_reg.predict(X_test)
```

### SVR
- C: Sum of Error Margins
- kernel:
  - linear: line of separation
  - rbf: circle of separation
    - Additional param gamma: Inverse of the radius
  - poly: curved line of separation
    - Additional param degree: Degree of the polynome
```python
from sklearn.svm import SVR
# Create an instance of the model
svr_reg = SVR(kernel="Linear",C=50)
# Fit the data
svr_reg.fit(X_train,y_train)
# Do predictions
svr_reg.predict(X_test)
```


## Clasification Algorithms

### Logistic Regression with stastmodel
parameters:
- β0,β1,β2,...,βn
```python
# Load the library
import statsmodels.api as sm
# Create an instance of the model
logitstat_clf = sm.Logit(y_train,X_train).fit()
# results
print(logitstat_clf.summary2())
# Do predictions
logitstat_clf.predict(X_test)
```

### Logistic Regression with sklearn
parameters:
- β0,β1,β2,...,βn
```python
# Load the library
from sklearn.linear_model import LogisticRegression
# Create an instance of the model
logitskl_clf = LogisticRegression()
# Fit the data
logitskl_clf.fit(X_train,y_train)
# Do predictions
logitskl_clf.predict(X_test)
```

### k nearest neighbor
parameters:
- n_neighbors
```python
# Import Library
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the model
knn_clf = KNeighborsClassifier(n_neighbors = 10)
# Fit the data
knn_clf.fit(X,y)
# Do predictions
knn_clf.predict(X_test)
```

### Decision Tree
parameters:
- Max_depth: Number of Splits
- Min_samples_leaf: Minimum number of observations per leaf
```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create an instance of the model
dcstree_clf = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
dcstree_clf.fit(X,y)
# Do predictions
dcstree_clf.predict(X_test)
```

### Gradient Boosted Trees
parameters:
- n_estimators: number of trees
- learning_rate: steps levels
- max_depth: tree depth
- min_samples_leaf: Minimum number of observations per leaf
```python
from sklearn.ensemble import GradientBoostingClassifier
# Create an instance of the model
gbm_clf = GradientBoostingClassifier(max_depth=4,n_estimators=100, learning_rate=0.1)
# Fit the data
gbm_clf.fit(X_train,y_train)
# Do predictions
gbm_clf.predict(X_test)
```

### Tree Forest
parameters:
- n_estimators: number of trees
- max_depth: tree depth
- min_samples_leaf: Minimum number of observations per leaf
```python
from sklearn.ensemble import RandomForestClassifier
# Create an instance of the model
rndf_clf = RandomForestClassifier(max_depth=4,n_estimators=100)
# Fit the data
rndf_clf.fit(X_train,y_train)
# Do predictions
rndf_clf.predict(X_test)
```

### SVC
Parameters:
- C: Sum of Error Margins
- kernel:
  - linear: line of separation
  - rbf: circle of separation
    - Additional param gamma: Inverse of the radius
  - poly: curved line of separation
    - Additional param degree: Degree of the polynome
```python
from sklearn.svm import SVC
# Create an instance of the model
svc_clf = SVC(kernel="Linear",C=50)
# Fit the data
svc_clf.fit(X_train,y_train)
# Do predictions
svc_clf.predict(X_test)
```

### Bayesian models
parameters:
- alpha:
```python
from sklearn.naive_bayes import BernoulliNB
# Create an instance of the model
bernoNB_clf = BernoulliNB(alpha=1.0e-10)
# Fit the data
bernoNB_clf.fit(X_train,y_train)
# Do predictions
bernoNB_clf.predict(X_test)
```

parameters:
```python
from sklearn.naive_bayes import GaussianNB
# Create an instance of the model
gaussNB_clf = GaussianNB()
# Fit the data
gaussNB_clf.fit(X_train,y_train)
# Do predictions
gaussNB_clf.predict(X_test)
```

parameters:
```python
from sklearn.naive_bayes import MultinomialNB
# Create an instance of the model
multNB_clf = MultinomialNB()
# Fit the data
multNB_clf.fit(X_train,y_train)
# Do predictions
multNB_clf.predict(X_test)
```


## Metrics

### Regression Metrics:
- Mean Absolute Error MAE
```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,y_pred)
```

- Mean Absolute Percentage Error MAPE
```python
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_true)) * 100
```

- Mean Squared Error MSE
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
```

- Root Mean Squared Error RMSE
```python
import numpy as np
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))
```

- R2 Score
```python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```

### Classification Metrics:
- Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
```

- Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
```

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
ax = sns.heatmap(confusion_matrix(y_test,y_pred) , annot=True, fmt="d",cmap="YlGnBu")
ax.set(xlabel='Predicted Values', ylabel='Actual Values',title='Confusion Matrix')
```

- Classification Reports
```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```

- Precision
```python
from sklearn.metrics import precision_score
precision_score(y_val, y_pred, pos_label='si')
```

- Recall
```python
from sklearn.metrics import recall_score
recall_score(y_val, y_pred, pos_label='si')
```

- f1 score
```python
from sklearn.metrics import f1_score
f1_score(y_val, y_pred, pos_label='si')
```

- roc curve

- auc

