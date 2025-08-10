## LINEAR REGRESSION


# 1. Import
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# 2. Create dataset
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + 5 + np.random.randn(100) * 2

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Instantiate model
model = LinearRegression()

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate (simple check)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")




## RANDOM FOREST CLASSIFIER




# 1. Import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 2. Create dataset
iris = load_iris()
X, y = iris.data, iris.target

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Instantiate model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")



## SVM




# 1. Import
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

# 2. Create dataset
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=1)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Instantiate model (with RBF kernel)
model = SVC(kernel='rbf', C=1.0)

# 5. Train model
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")



