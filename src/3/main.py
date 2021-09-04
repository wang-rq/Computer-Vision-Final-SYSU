import os
import numpy as np 
from segment_graph import *
from rgb_feature import *
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config import *
from full_feature import *
from sklearn import svm


def classify():

    x_train, y_train = np.load(train_file_path+"x_train_full.npy"), np.load(train_file_path+"y_train.npy")
    x_test, y_test = np.load(test_file_path+"x_test_full.npy"), np.load(test_file_path+"y_test.npy")

    print("data loaded...\nstart classifying...")
    # model = RandomForestClassifier()
    model = svm.SVC()
    model.fit(x_train, y_train)
    y_test_predict = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_test_predict))


if __name__ == '__main__':
    generate_rgb_feature()
    generate_full_feature()
    classify()