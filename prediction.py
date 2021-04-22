# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
from PIL import Image
import pickle
import random
import os
import time



def prediction(s):
	# load the VGG16 network and initialize the label encoder
	print("[INFO] loading network...")


	path = "image_as_array_02_24_2021\\%s\\"%s
	image_total = np.load(path + "image_total.npy")
	model = VGG16(weights="imagenet", include_top=False)

	model = VGG16(weights="imagenet", include_top=False)
	le = None

	labels = []
	labels += (["1"] * len(image_total))

	# if the label encoder is None, create it
	if le is None:
		le = LabelEncoder()
		le.fit(labels)

	# imageName = []
	# with open(csvPath, "w") as csv:
	data = []
	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(image_total), 32)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("[INFO] processing batch {}/{}".format(b + 1,
			int(np.ceil(len(image_total) / float(32)))))
		batchPaths = image_total[i:i + 32]
		batchLabels = le.transform(labels[i:i + 32])
		batchImages = []

		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility
			# while ensuring the image is resized to 224x224 pixels
			image = Image.fromarray(imagePath)
			image = image.resize(size=(224, 224))
			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)

			# imageName.append(imagePath)

		# pass the images through the network and use the outputs as
		# our actual features, then reshape the features into a
		# flattened volume
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=32)
		features = features.reshape((features.shape[0], 7 * 7 * 512))



		# loop over the class labels and extracted features
		for (label, vec) in zip(batchLabels, features):
			# construct a row that exists of the class label and
			# extracted features
			# vec = ",".join([str(v) for v in vec])
			vec = np.array(vec, dtype="float")
			data.append(vec)
			# csv.write("{},{}\n".format(label, vec))
	print("Done feature extraction")

	# train the model
	print("[INFO] training model...")
	with open("model.cpickle",'rb') as fp:
	    model = pickle.load(fp)
	    print("[INFO] evaluating...")
	    data = np.array(data)
	    # val = model.predict(data)
	    probabilities = model.predict_proba(data)
	result = []
	for i in probabilities:
		if (i[0] < 0.5):
			result.append("1")
		elif (i[0] > 0.95):
			result.append("0")
		else:
			result.append("2")

	np.save("image_as_array_02_24_2021\\result\\" + "predict_result_%s.npy"%s, np.array(result))

if __name__ == '__main__':
	dirlist = os.listdir("image_as_array_02_24_2021")
	for s in dirlist:
		print("---------------------------------------------------------------------")
		print(s + " is Begin")
		print("---------------------------------------------------------------------")
		prediction(s)
		print("---------------------------------------------------------------------")
		print(s + " is done")
		print("---------------------------------------------------------------------")