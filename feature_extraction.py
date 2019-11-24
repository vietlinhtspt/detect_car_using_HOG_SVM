import numpy as np
import cv2
import scipy
from skimage.feature import hog
from config import feat_extraction_params


def get_hog_features(img, orient, pix_per_cell, cell_per_block, verbose=True, feature_vec=True):
    features, hog_image = hog(img,
                              orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block),
                              transform_sqrt=True,
                              visualize=verbose,
                              feature_vector=feature_vec)
    return features, hog_image


def binned_features(img, size=(32, 32)):
    """
    Return binned color features is the resized image.
    """
    binned_features = cv2.resize(img, size).ravel()
    return binned_features


def color_histogram_features(img, nbins=32, bins_range=(0, 256)):
    """
    Compute the color histogram features of a given image `img`.
    Histogram is computed for each channel.
    Return histogram features is value after concatenate all histograms 
    """
    channel1_histogram = np.histogram(
        img[:, :, 0], bins=nbins, range=bins_range)
    channel2_histogram = np.histogram(
        img[:, :, 1], bins=nbins, range=bins_range)
    channel3_histogram = np.histogram(
        img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    histogram_features = np.concatenate(
        (channel1_histogram[0], channel2_histogram[0], channel3_histogram[0]))
    return histogram_features


def image_to_features(image, feat_extraction_params):
    """
    Extract and return the feature vector from given image.
    Parameters
    ----------
    image : ndarray
        input image on which perform feature extraction.
    feat_extraction_params : dict
        dictionary of parameters that control the process of feature extraction.
    Returns
    -------
    features : ndarray
        array of features which describes the input image.
    """
    color_space = feat_extraction_params['color_space']
    orient = feat_extraction_params['orient']
    pix_per_cell = feat_extraction_params['pix_per_cell']
    cell_per_block = feat_extraction_params['cell_per_block']
    hog_channel = feat_extraction_params['hog_channel']
    spatial_size = feat_extraction_params['spatial_size']
    hist_bins = feat_extraction_params['hist_bins']

    # Can converse channel color in here, variable feature using store new image.
    feature_image = np.copy(image)

    image_features = []

    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_feature, hog_image = get_hog_features(feature_image[:, :, channel],
                                                  orient, pix_per_cell, cell_per_block,
                                                  verbose=True, feature_vec=True)
        hog_features.append(hog_feature)
    hog_features = np.ravel(hog_features)

    # # Get color features
    # color_features = binned_features(feature_image, size=spatial_size)
    # histogram_features = color_histogram_feature(
    #     feature_image, nbins=hist_bins)

    # # Concatenate all features to a singer vector
    # features = np.hstack(
    #     (color_features, histogram_features, hog_features)).reshape(-1)

    # return features
    return hog_features


def extract_features_from_file_list(file_list, feat_extraction_params):
    """
    Extract features from a list of images
    Parameters
    ----------
    file_list : list
        list of files path on which feature extraction process must be performed.
    feat_extraction_params : dict
        dictionary of parameters that control the process of feature extraction.
    Returns
    -------
    features : list
        list of feature array, one for each input file.
    """
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of image files
    for file in file_list:

        resize_h, resize_w = feat_extraction_params['resize_h'], feat_extraction_params['resize_w']
        image = cv2.resize(cv2.imread(file), (resize_w, resize_h))

        # compute the features of this particular image, then append to the list
        file_features = image_to_features(image, feat_extraction_params)
        features.append(file_features)

    return features


if __name__ == '__main__':

    resize_w, resize_h = 64, 64
    img = cv2.resize(cv2.imread(
        "./data/vehicles/Far/image0000.png"), (resize_w, resize_h))
    feature = image_to_features(img, feat_extraction_params)
    print(feature)
    print("Extract done")
