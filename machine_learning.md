# Machine Learning Snippets

## GridSearch
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
neigh_reg = GridSearchCV(KNeighborsRegressor(),param_grid={"n_neighbors":np.arange(3,50)},cv = 10,scoring = "mean_squared_error",n_jobs=-1)
# Fit will test all of the combinations
neigh_reg.fit(X,y)
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
dcstree_clf.fit(X,y)
# Best estimator and best parameters
dcstree_clf.best_estimator_
dcstree_clf.best_params_
dcstree_clf.best_score_
```


## Regression

### Linear Regression
```python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg = LinearRegression()
# Fit the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[2540],[3500],[4000]])
```

### k Nearest Neighbors
parameters:
- n_neighbors
```python
# Load the library
from sklearn.neighbors import KNeighborsRegressor
# Create an instance
regk = KNeighborsRegressor(n_neighbors=2)
# Fit the data
regk.fit(X,y)
```

### Decision Tree
parameters:
- Max_depth: Number of Splits
- Min_samples_leaf: Minimum number of observations per leaf
```python
# Load the library
from sklearn.tree import DecisionTreeRegressor
# Create an instance
regd = DecisionTreeRegressor(max_depth=3)
# Fit the data
regd.fit(X,y)
```


## Clasification

### Logistic Regression
```python
# Load the library
from sklearn.linear_model import LogisticRegression
# Create an instance of the classifier
clf = LogisticRegression()
# Fit the data
clf.fit(X,y)
```

### k nearest neighbor
parameters:
- n_neighbors
```python
# Import Library
from sklearn.neighbors import KNeighborsClassifier
# Create instance
clfk = KNeighborsClassifier(n_neighbors = 5)
# Fit
clfk.fit(X,y)
```

### SVM
Parameters:
- C: Sum of Error Margins
- kernel:
  - linear: line of separation
  - rbf: circle of separation
    - Additional param gamma: Inverse of the radius
  - poly: curved line of separation
    - Additional param degree: Degree of the polynome
```python
# Import Library
from sklearn.svm import SVC
# Create instance
clfsvm = SVC(kernel = "linear",C = 10)
# Fit
clfsvm.fit(X,y)
```
### Decision Tree
parameters:
- Max_depth: Number of Splits
- Min_samples_leaf: Minimum number of observations per leaf
```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create instance
clfd = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
clfd.fit(X,y)
```
