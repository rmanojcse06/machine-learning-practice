from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X,y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    n_informative=2,
    random_state=42
)
print("X=",X)
print("y=",y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=None, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))