from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from datetime import datetime
import _pickle as cPickle
import numpy as np
import random
import time
import os

# get all the classifiers we want to experiment with
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def train_and_test_statistical_model(name, classifier, x_train, y_train, x_test, y_test):
    # fit the model on your dataset
    trained_classifier = classifier.fit(x_train, y_train)
    # get predictions on unseen data
    y_pred = trained_classifier.predict(x_test)
    # get an accuracy score please
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('Model Accuracy: {:.2f}%'.format(100*accuracy))
    # get confusion matrix
    confusion_matrix_to_print = confusion_matrix(y_test, y_pred)
    # show confusion matrix and classification report for precision, recall, F1-score
    print("################################ {} ################################".format(name))
    print('Confusion Matrix')
    print(confusion_matrix_to_print)
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=['Null-Pixels', 'Non-Forest', 'Forest']))
    return trained_classifier


if __name__ == "__main__":
    # raw_dataset_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Clipped dataset\\Pickled_data\\"
    # processed_dataset_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Clipped dataset\\statistical_models_dataset\\1M_dataset.pkl"
    # model_path = "E:\\Forest Cover - Redo 2020\\Google Cloud - Training\\Training Data\\Clipped dataset\\statistical_models_dataset\\logistic_regressor.pkl"

    raw_dataset_path = "/home/azulfiqar_bee15seecs/training_data/pickled_clipped_training_data"
    processed_dataset_path = "/home/azulfiqar_bee15seecs/statistical_models_dataset/1M_dataset.pkl"
    model_path = "/home/azulfiqar_bee15seecs/statistical_models_dataset/logistic_regressor.pkl"

    # get your model (RandomForestClassifier, DecisionTreeClassifier, SVC, GaussianNB, LogisticRegression, Perceptron)
    classifiers = {
        "SVC": SVC(verbose=1),
        "Perceptron": Perceptron(verbose=1, n_jobs=4),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": LogisticRegression(verbose=1, n_jobs=4, max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(verbose=1, n_jobs=4),
    }
    # prepare data if it doesn't already exist
    if os.path.exists(processed_dataset_path):
        print("(LOG): Found Precompiled Serialized Dataset...")
        with open(processed_dataset_path, 'rb') as processed_dataset:
            (datapoints_as_array, labels_as_array) = cPickle.load(processed_dataset)
        print("(LOG): Dataset Size: Datapoints = {}; Ground Truth Labels {}".format(datapoints_as_array.shape, labels_as_array.shape))
        print("(LOG): Loaded Precompiled Serialized Dataset Successfully!")
    else:
        print("(LOG): No Precompiled Dataset Found! Creating New Dataset Now...")
        all_pickle_files_in_pickled_dataset = [os.path.join(raw_dataset_path, x) for x in os.listdir(raw_dataset_path)]
        datapoints_as_array, labels_as_array = np.empty(shape=[1,18]), np.empty(shape=[1,])
        # random.seed(datetime.now())
        np.random.seed(0)
        for idx, this_pickled_file in enumerate(all_pickle_files_in_pickled_dataset):
            if idx % 100 == 0:
                print("(LOG): Processing ({}/{}) => {}".format(idx, len(all_pickle_files_in_pickled_dataset), this_pickled_file))
            with open(this_pickled_file, 'rb') as this_small_data_sample:
                small_image_sample, small_label_sample = cPickle.load(this_small_data_sample)
                this_shape = small_image_sample.shape
                random_rows, random_cols = np.random.randint(0, this_shape[0], size=606), np.random.randint(0, this_shape[0], size=606)
                sample_datapoint = small_image_sample[random_rows, random_cols, :]
                sample_label = small_label_sample[random_rows, random_cols]
                # get more indices to add to the example, landsat-8
                ndvi_band = (sample_datapoint[:, 4]-sample_datapoint[:, 3]) / (sample_datapoint[:, 4] + sample_datapoint[:, 3] + 1e-7)
                evi_band = 2.5 * (sample_datapoint[:, 4] - sample_datapoint[:, 3]) / (sample_datapoint[:, 4] + 6 * sample_datapoint[:, 3] - 7.5 * sample_datapoint[:, 1] + 1)
                savi_band = 1.5 * (sample_datapoint[:, 4] - sample_datapoint[:, 3]) / (sample_datapoint[:, 4] + sample_datapoint[:, 3] + 0.5)
                msavi_band = 0.5 * (2 * sample_datapoint[:, 4] + 1 - np.sqrt((2 * sample_datapoint[:, 4] + 1) ** 2 - 8 * (sample_datapoint[:, 4] - sample_datapoint[:, 3])))
                ndmi_band = (sample_datapoint[:, 4] - sample_datapoint[:, 5]) / (sample_datapoint[:, 4] + sample_datapoint[:, 5] + 1e-7)
                nbr_band = (sample_datapoint[:, 4] - sample_datapoint[:, 6]) / (sample_datapoint[:, 4] + sample_datapoint[:, 6] + 1e-7)
                nbr2_band = (sample_datapoint[:, 5] - sample_datapoint[:, 6]) / (sample_datapoint[:, 5] + sample_datapoint[:, 6] + 1e-7)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(ndvi_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(evi_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(savi_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(msavi_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(ndmi_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(nbr_band, axis=1)), axis=1)
                sample_datapoint = np.concatenate((sample_datapoint, np.expand_dims(nbr2_band, axis=1)), axis=1)
            datapoints_as_array = np.concatenate((datapoints_as_array, sample_datapoint), axis=0)
            labels_as_array = np.concatenate((labels_as_array, sample_label), axis=0)
            # at this point, we just serialize the arrays and save them
            with open(processed_dataset_path, 'wb') as processed_dataset:
                cPickle.dump((datapoints_as_array, labels_as_array), processed_dataset)
        print("(LOG): Dataset Size: Datapoints = {}; Ground Truth Labels {}".format(datapoints_as_array.shape, labels_as_array.shape))
        print("(LOG): Compiled and Serialized New Dataset Successfully!")
        pass
    # create training and testing arrays from loaded data
    total_datapoints = len(datapoints_as_array)
    split = int(0.8*total_datapoints)
    x_train, y_train = datapoints_as_array[:split], labels_as_array[:split].astype(np.uint8)
    x_test, y_test = datapoints_as_array[split:], labels_as_array[split:].astype(np.uint8)
    print("(LOG): Dataset for Training and Testing Prepared")
    print("(LOG): Training Data: {}; Testing Data: {}".format(x_train.shape, x_test.shape))
    # call model for training
    trained_classifier = train_and_test_statistical_model(name="LogisticRegression", classifier=classifiers["LogisticRegression"],
                                                          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    with open(model_path, 'wb') as model_file:
        cPickle.dump(trained_classifier, model_file)
    print("(LOG): Saved Trained Classifier as {}".format(model_path))

    # basic test
    # # load a test dataset and split it
    # dataset = load_wine()
    # X, Y = dataset.data, dataset.target
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    # print("Previous Dataset Size: ", x_train.shape)
    # for _ in range(13):
    #     x_train = np.concatenate((x_train, x_train), axis=0)
    #     y_train = np.concatenate((y_train, y_train), axis=0)
    # print("Augmented Dataset Size: ", x_train.shape)
    # train_and_test_statistical_model(name="LogisticRegression", classifier=classifiers["LogisticRegression"])
    # for name, this_classifier in classifiers.items():
    #     run(name=name, classifier=this_classifier)
