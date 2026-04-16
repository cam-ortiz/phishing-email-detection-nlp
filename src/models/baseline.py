from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Trains the logistic regression model
def train_logistic_regression(X, y) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# Trains the Naive Bayes model
def train_naive_bayes(X, y) -> MultinomialNB:
    model = MultinomialNB()
    model.fit(X, y)
    return model


# Trains the linear SVM model
def train_svm(X, y) -> LinearSVC:
    model = LinearSVC(max_iter=1000)
    model.fit(X, y)
    return model


# Return accuracy, precision, recall, and F1 for a trained model on test data.
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
