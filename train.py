# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import pickle
import random
import os
import time

start = time.time()
 
# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None

# loop over the data splits
split = "samples"
	# grab all image paths in the current split
print("[INFO] processing '{} split'...".format(split))
imagePaths = list(paths.list_images(split))

# randomly shuffle the image paths and then extract the class
# labels from the file paths
random.shuffle(imagePaths)

labels = []
labels += (["1"] * len(imagePaths))

# if the label encoder is None, create it
if le is None:
	le = LabelEncoder()
	le.fit(labels)

# open the output CSV file for writing
csvPath = os.path.sep.join(["profile", "{}.csv".format(split)])


# imageName = []
# with open(csvPath, "w") as csv:
data = []
# loop over the images in batches
for (b, i) in enumerate(range(0, len(imagePaths), 32)):
	# extract the batch of images and labels, then initialize the
	# list of actual images that will be passed through the network
	# for feature extraction
	print("[INFO] processing batch {}/{}".format(b + 1,
		int(np.ceil(len(imagePaths) / float(32)))))
	batchPaths = imagePaths[i:i + 32]
	batchLabels = le.transform(labels[i:i + 32])
	batchImages = []

	# loop over the images and labels in the current batch
	for imagePath in batchPaths:
		# load the input image using the Keras helper utility
		# while ensuring the image is resized to 224x224 pixels
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)
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
# load the label encoder from disk
# le = pickle.loads(open(config.LE_PATH, "rb").read())

# train the model
print("[INFO] training model...")
with open("profile\\model.cpickle",'rb') as fp:
    model = pickle.load(fp)
# evaluate the model
    print("[INFO] evaluating...")
    data = np.array(data)
    val = model.predict(data)
    probabilities = model.predict_proba(data)

# for i in range(len(val)):
#     if (int(val[i]) == 1):
#         val[i] = 0
#     else:
#         val[i] = 1

# with open("wrong.csv", "w") as csv:
#     csv.write("{},{},{},{}, {}\n".format("image path", "manual label result","model prediction","probabilities for forliage", "probabilities for gap"))
#     for i in range(len(val)):
#         label = imagePaths[i].split("_")[3]
#         if (int(val[i]) != int(label)):
#             csv.write("{},{},{},{}, {}\n".format(imagePaths[i],label,val[i], probabilities[i][0], probabilities[i][1]))


# with open("correct.csv", "w") as csv:
#     csv.write("{},{},{},{}, {}\n".format("image path", "manual label result","model prediction","probabilities for forliage", "probabilities for gap"))
#     for i in range(len(val)):
#         label = imagePaths[i].split("_")[3]
#         if (int(val[i]) == int(label)):
#             csv.write("{},{},{},{}, {}\n".format(imagePaths[i],label,val[i], probabilities[i][0], probabilities[i][1]))
# end = time.time()
# print(end - start)
# print(val)