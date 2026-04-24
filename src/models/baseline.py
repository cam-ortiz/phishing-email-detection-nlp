from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


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
