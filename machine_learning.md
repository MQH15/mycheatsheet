# Machine Learning Snippets

## GridSearch
```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
reg_test = GridSearchCV(KNeighborsRegressor(),param_grid={"n_neighbors":np.arange(3,50)})
# Fit will test all of the combinations
reg_test.fit(X,y)
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
```python
# Import Library
from sklearn.neighbors import KNeighborsClassifier
# Create instance
clfk = KNeighborsClassifier(n_neighbors = 5)
# Fit
clfk.fit(X,y)
```

### SVM
```python
# Import Library
from sklearn.svm import SVC
# Create instance
clfsvm = SVC(kernel = "linear",C = 10)
# Fit
clfsvm.fit(X,y)
```
### Decision Tree
```python
# Import library
from sklearn.tree import DecisionTreeClassifier
# Create instance
clfd = DecisionTreeClassifier(min_samples_leaf=20,max_depth=3)
# Fit the data
clfd.fit(X,y)
```
