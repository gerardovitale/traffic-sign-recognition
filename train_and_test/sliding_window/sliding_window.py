import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import imutils
import time

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from imutils.object_detection import non_max_suppression
import tensorflow as tf

from train_and_test.sliding_window.constant_variables import WIDTH, SCALE, STEP, WINDOW_SIZE, INPUT_SIZE, MODEL_PATH, CLASS_ID, CLASS_NAMES

def load_image(image_path, resizing=False, width=WIDTH):
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if resizing:
		image = imutils.resize(image, width=WIDTH)
	return image

def scale_image(image, scale=2, minSize=(28, 28)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
	yield image

def sliding_window(image, step_size, window_size):
	# slide a window across the image
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# yield the current window
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def model_predictions(y, model_path=MODEL_PATH, verbose=0):
	model = load_model(model_path)
	preds = model.predict(y, verbose)
	return preds

def app(image, window_size=(28,28), vizualize=False):
	location = []
	rois = []
	winW, winH = window_size
	(H, W) = image.shape[:2]
	# loop over the image pyramid
	start = time.time()
	for resized in scale_image(image, scale=1.5):
		print('[INFO] Another scale')
		scale = W / float(resized.shape[1])
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, step_size=32, window_size=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			x = int(x * scale)
			y = int(y * scale)
			w = int(winW * scale)
			h = int(winH * scale)
			# take the ROI and preprocess it so we can later classify
			# the region using Keras/TensorFlow
			roi = cv2.resize(window, INPUT_SIZE)
			roi = img_to_array(roi)
			roi = preprocess_input(roi)
			# update our list of ROIs and associated coordinates
			rois.append(roi)
			location.append((x, y, x + w, y + h))
			if vizualize:
				# clone the original image and then draw a bounding box
				# surrounding the current region
				clone = image.copy()
				cv2.rectangle(
					clone, 
					(x, y), 
					(x + w, y + h),
					(0, 255, 0), 
					2
				)
				# show the visualization and current ROI
				plt.imshow(clone)
				plt.imshow(window)
				cv2.waitKey(0)
				plt.show()
	# show how long it took to loop over the image pyramid layers and
	# sliding window locations
	end = time.time()
	print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(end - start))
	# convert the ROIs to a NumPy array
	rois = np.array(rois, dtype="float32")
	# classify each of the proposal ROIs using ResNet and then show how
	# long the classifications took
	print("[INFO] classifying ROIs...")
	start = time.time()
	preds = model_predictions(rois, model_path=MODEL_PATH, verbose=0)
	end = time.time()
	print("[INFO] classifying ROIs took {:.5f} seconds".format(end - start))
	# decode the predictions and initialize a dictionary which maps class
	# labels (keys) to any ROIs associated with that label (values)
	# preds = imagenet_utils.decode_predictions(preds, top=1)
	labels = {}

	# loop over the predictions
	for (i, p) in enumerate(preds):
		# grab the prediction information for the current ROI
		prob = np.max(tf.nn.softmax(p))
		label = CLASS_NAMES[CLASS_ID[np.argmax(p)]]
		# filter out weak detections by ensuring the predicted probability
		# is greater than the minimum probability
		if prob >= 0.999:
			# grab the bounding box associated with the prediction and
			# convert the coordinates
			box = location[i]
			# grab the list of predictions for the label and add the
			# bounding box and probability to the list
			L = labels.get(label, [])
			L.append((box, prob))
			labels[label] = L
	# loop over the labels for each of detected objects in the image
	for label in labels.keys():
		# clone the original image so that we can draw on it
		print("[INFO] showing results for '{}'".format(label))
		clone = image.copy()
		# loop over all bounding boxes for the current label
		for (box, prob) in labels[label]:
			# draw the bounding box on the image
			(startX, startY, endX, endY) = box
			cv2.rectangle(
				clone, 
				(startX, startY), 
				(endX, endY),
				(0, 255, 0), 
				2
			)
		# show the results *before* applying non-maxima suppression, then
		# clone the image again so we can display the results *after*
		# applying non-maxima suppression
		plt.imshow(clone)
		plt.show()
		clone = image.copy()

		# extract the bounding boxes and associated prediction
		# probabilities, then apply non-maxima suppression
		boxes = np.array([p[0] for p in labels[label]])
		proba = np.array([p[1] for p in labels[label]])
		boxes = non_max_suppression(boxes, proba)
		# loop over all bounding boxes that were kept after applying
		# non-maxima suppression
		for (startX, startY, endX, endY) in boxes:
			# draw the bounding box and label on the image
			cv2.rectangle(clone, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.putText(
				clone, 
				label, 
				(startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 
				0.45, 
				(0, 255, 0),
				2)
		# show the output after apply non-maxima suppression
		plt.imshow(clone)
		plt.show()
		cv2.waitKey(0)

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")