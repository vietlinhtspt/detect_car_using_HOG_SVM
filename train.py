import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from computer_vision_utils.filesystem import get_file_list_recursively
from config import root_data_non_vehicle, root_data_vehicle, feat_extraction_params
from feature_extraction import extract_features_from_file_list

if __name__ == '__main__':
    # Read all paths of training images
    cars = get_file_list_recursively(root_data_vehicle)
    notcars = get_file_list_recursively(root_data_non_vehicle)
    # Preprocessing data(check training time)
    # Get feature using HOG
    t = time.time()
    print('Extracting car features...')
    car_features = extract_features_from_file_list(
        cars, feat_extraction_params)
    print('Extracting non-car features...')
    notcar_features = extract_features_from_file_list(
        notcars, feat_extraction_params)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract feature...')

    # Concatenate trainning data positive and negative
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    print("X: ", X.shape)
    # standardize features with sklearn preprocessing
    feature_scaler = StandardScaler().fit(X)
    scaled_X = feature_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[1]))

    # Define the classifier
    svc = LinearSVC()

    # Train the classifier (check training time)
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # dump all stuff necessary to perform testing in a successive phase
    with open('data/svm_trained.pickle', 'wb') as f:
        pickle.dump(svc, f)
    with open('data/feature_scaler.pickle', 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open('data/feat_extraction_params.pickle', 'wb') as f:
        pickle.dump(feat_extraction_params, f)
