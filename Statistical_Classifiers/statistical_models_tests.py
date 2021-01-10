from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine

# all the classifiers we want to try here
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def run(name, classifier):
    # load a test dataset and split it
    dataset = load_wine()
    X, Y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    # fit the model on your dataset
    trained_classifier = classifier.fit(X_train, y_train)
    # get predictions on unseen data
    y_pred = trained_classifier.predict(X_test)
    # get confusion matrix
    confusion_matrix_to_print = confusion_matrix(y_test, y_pred)
    # show confusion matrix and classification report for precision, recall, F1-score
    print("################################ {} ################################".format(name))
    print('Confusion Matrix\n')
    print(confusion_matrix_to_print)
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['Null-Pixels', 'Non-Forest', 'Forest']))
    pass


if __name__ == "__main__":
    # get your model (RandomForestClassifier, DecisionTreeClassifier, SVC, GaussianNB, LogisticRegression, Perceptron)
    classifiers = {
        "SVC": SVC(),
        "Perceptron": Perceptron(),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
    }
    for name, this_classifier in classifiers.items():
        run(name=name, classifier=this_classifier)
        break
