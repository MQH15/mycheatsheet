# Machine Learning Snippets

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

## Regression Algorithms

### Linear Regression with stastmodel
```python
# Load the library
import statsmodels.formula.api as smf
# Create an instance of the model
linearstat_reg = smf.ols(formula="Sales~TV+Radio", data = datatraining).fit()
# results
linearstat_reg.summary()
# Do predictions
linearstat_reg.predict(datatesting)
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

### Support Vector
parameters:
- kernel: flat type
- C: number of points in the margin
```python
from sklearn.svm import SVR
# Create an instance of the model
svr_reg = SVR(kernel="Linear",C=50)
# Fit the data
svr_reg.fit(X_train,y_train)
# Do predictions
svr_reg.predict(X_test)
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
