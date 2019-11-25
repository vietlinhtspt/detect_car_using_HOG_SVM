import numpy as np
import cv2
import os
import pickle
import scipy
import time
import matplotlib.pyplot as plt
from feature_extraction import image_to_features, binned_features, color_histogram_features, get_hog_features
# generate random integer values
from random import seed
from random import randint
import matplotlib.pyplot as plt


def normalize_image(img):

    img = np.float32(img)

    img = img / img.max() * 255

    return np.uint8(img)


def draw_bouding_boxes(frame, labeled, num_cars):

    for car_number in range(1, num_cars+1):

        rows, cols = np.where(labeled == car_number)

        # Find minimum enclosing rectangle
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      color=(255, 0, 0), thickness=6)

    return frame


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    x_span = x_start_stop[1] - x_start_stop[0]
    y_span = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    n_x_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    n_y_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x / y
    n_x_windows = np.int(x_span / n_x_pix_per_step) - 1
    n_y_windows = np.int(y_span / n_y_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions.
    for i in range(n_y_windows):
        for j in range(n_x_windows):
            # Calculate window position
            start_x = j * n_x_pix_per_step + x_start_stop[0]
            end_x = start_x + xy_window[0]
            start_y = i * n_y_pix_per_step + y_start_stop[0]
            end_y = start_y + xy_window[1]

            # Append window position to list
            window_list.append(((start_x, start_y), (end_x, end_y)))

    # Return the list of windows
    return window_list


def draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6):

    # Make a copy of the image
    img_copy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bbox_list:
        # Draw a rectangle given bbox coordinates
        tl_corner = tuple(bbox[0])
        br_corner = tuple(bbox[1])
        cv2.rectangle(img_copy, tl_corner, br_corner, color, thick)

    # Return the image copy with boxes drawn
    return img_copy


# Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, feat_extraction_params):

    hot_windows = []  # list to receive positive detection windows
    images = []

    for window in windows:
        # Extract the current window from original image
        resize_h, resize_w = feat_extraction_params['resize_h'], feat_extraction_params['resize_w']
        test_img = cv2.resize(
            img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (resize_w, resize_h))

        # Extract features for that window using single_img_features()
        features = image_to_features(test_img, feat_extraction_params)

        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # Predict on rescaled features
        prediction = clf.predict(test_features)

        # If positive (prediction == 1) then save the window
        if prediction == 1:
            images.append(test_img)
            hot_windows.append(window)

    # Return windows for positive detections
    return hot_windows, images


def compute_heatmap_from_detections(frame, hot_windows, threshold=5, verbose=False):

    h, w = frame.shape

    heatmap = np.zeros(shape=(h, w), dtype=np.uint8)

    for bbox in hot_windows:
        # for each bounding box, add heat to the corresponding rectangle in the image
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[y_min:y_max, x_min:x_max] += 1  # add heat

    # Apply threshold
    ret, heatmap_thresh = cv2.threshold(
        heatmap, threshold, 255, type=cv2.THRESH_BINARY)

    # Apply morphological closure to remove noise
    heatmap_thresh = cv2.morphologyEx(heatmap_thresh, op=cv2.MORPH_CLOSE,
                                      kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)

    return heatmap, heatmap_thresh


def find_cars(image, y_start, y_stop, scale, svc, feature_scaler, feature_extraction_params):

    hot_windows = []

    resize_h = feature_extraction_params['resize_h']
    resize_w = feature_extraction_params['resize_w']
    spatial_size = feature_extraction_params['spatial_size']
    hist_bins = feature_extraction_params['hist_bins']
    orient = feature_extraction_params['orient']
    pix_per_cell = feature_extraction_params['pix_per_cell']
    cell_per_block = feature_extraction_params['cell_per_block']

    draw_img = np.copy(image)

    image_crop = image[y_start:y_stop, :]

    if scale != 1:
        imshape = image_crop.shape
        image_crop = cv2.resize(
            image_crop, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # Define blocks and steps as above
    n_x_blocks = (image_crop.shape[1] // pix_per_cell) - 1
    n_y_blocks = (image_crop.shape[0] // pix_per_cell) - 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 8  # Instead of overlap, define how many cells to step
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    hog, _ = get_hog_features(image_crop, orient, pix_per_cell,
                              cell_per_block, feature_vec=False)

    for xb in range(n_x_steps):
        for yb in range(n_y_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step

            hog_features = hog[y_pos:y_pos + n_blocks_per_window,
                               x_pos:x_pos + n_blocks_per_window].ravel().reshape(1, -1)
            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            test_prediction = svc.predict(hog_features)

            if test_prediction == 1:
                xbox_left = np.int(x_left * scale)
                ytop_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                tl_corner_draw = (xbox_left, ytop_draw + y_start)
                br_corner_draw = (xbox_left + win_draw,
                                  ytop_draw + win_draw + y_start)

                cv2.rectangle(draw_img, tl_corner_draw,
                              br_corner_draw, (0, 0, 255), 6)

                hot_windows.append((tl_corner_draw, br_corner_draw))

    return hot_windows


def detect_car(frame, svc, feature_scaler, feat_extraction_params, keep_state=False, verbose=False):

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    hot_windows = []

    for subsample in np.arange(1, 2):
        hot_windows += find_cars(frame, 152, 280, subsample,
                                 svc, feature_scaler, feat_extraction_params)

    # compute heatmaps positive windows found
    thresh = (time_window - 1) if keep_state else 0
    heatmap, heatmap_thresh = compute_heatmap_from_detections(
        frame, hot_windows, threshold=thresh, verbose=False)

    # label connected components
    labeled_frame, num_objects = scipy.ndimage.measurements.label(
        heatmap_thresh)

    return labeled_frame, num_objects


if __name__ == '__main__':
    # # load model
    svc = pickle.load(open('./data/svm_trained.pickle', 'rb'))
    feature_scaler = pickle.load(open('./data/feature_scaler.pickle', 'rb'))
    feat_extraction_params = pickle.load(
        open('./data/feat_extraction_params.pickle', 'rb'))

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('video1.mp4')

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            t = time.time()
            print(frame.shape)

            labeled_frame, num_objects = detect_car(frame, svc, feature_scaler,
                                                    feat_extraction_params, verbose=False)

            if num_objects != 0:
                img_detection = draw_bouding_boxes(
                    frame.copy(), labeled_frame, num_objects)        # draw detected bboxes

                name = str(randint(0, 1000000000)) + '.jpg'
                out_path = './output_images/' + name
                cv2.imwrite(out_path, img_detection)

            print('Done. Elapsed: {:.02f}'.format(time.time()-t))

        # Break the loop
        else:
            break
