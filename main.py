from build_dataset import build_dataset
from pyimagesearch import config
from extract_features import extract_features
from shutil import move,rmtree,make_archive
from train import train
import numpy as np
import shutil
import os
import time


 
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "ForG"
 
# initialize the base path to the *new* directory that will contain
# # # our images after computing the training and testing split\
# BASE = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\original\\original_right"

# BASE_list = ["wl_340\\wl_340_ol_0"
# 			,"wl_380\\wl_380_ol_100","wl_380\\wl_380_ol_75","wl_380\\wl_380_ol_50","wl_380\\wl_380_ol_25","wl_380\\wl_380_ol_0"
# 			,"wl_400\\wl_400_ol_100","wl_400\\wl_400_ol_75","wl_400\\wl_400_ol_50","wl_400\\wl_400_ol_25","wl_400\\wl_400_ol_0"
# 			,"wl_420\\wl_420_ol_100","wl_420\\wl_420_ol_75","wl_420\\wl_420_ol_50","wl_420\\wl_420_ol_25","wl_420\\wl_420_ol_0"
# 			,"wl_440\\wl_440_ol_100","wl_440\\wl_440_ol_75","wl_440\\wl_440_ol_25","wl_440\\wl_440_ol_0"]


# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"
 
# initialize the list of class label names
CLASSES = ["gap", "foliage", "unsure"]
 
# set the batch size
BATCH_SIZE = 16

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
 
# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])





index = np.load("e_t_index.npy", allow_pickle=True)

BASE_list = []
set_list = os.listdir("F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\WL")



for s in set_list:
	ol_list = os.listdir("F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\WL\\" + s)
	for i in ol_list:
		BASE_list.append("WL\\"+s+"\\"+i)




# BASE_list = ["WL\\wl_1960\\wl_1960_ol_50"]




for i in range(len(BASE_list)):
	BASE = BASE_list[i]
	deliminator = "_"
	name_base = BASE.split("\\")[2]

	BASE = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\" + BASE
	BASE_PATH = BASE + "\\" + name_base
	print(BASE_PATH)
	s = time.time()
	
	os.chdir(BASE)
	


	if not os.path.exists(ORIG_INPUT_DATASET):
	    os.makedirs(ORIG_INPUT_DATASET)
	if not os.path.exists(BASE_PATH):
	    os.makedirs(BASE_PATH)
	if not os.path.exists(ORIG_INPUT_DATASET + "\\evaluation"):
	    os.makedirs(ORIG_INPUT_DATASET + "\\evaluation")
	if not os.path.exists(ORIG_INPUT_DATASET + "\\training"):
	    os.makedirs(ORIG_INPUT_DATASET + "\\training")
	if not os.path.exists("output"):
	    os.makedirs("output")


	# print("Random Classify ... ")
	# # Random selective
	# image_list = os.listdir()
	# image_list = [str(i) for i in image_list if '.jpg' in str(i)]
	# image_list = [str(i) for i in image_list if i.split("_")[0] != "2"]
	# total = len(image_list)

	# x = np.random.choice(range(total), total, replace=False)
	# e = int(total / 5)

	# for i in x[:e]:
	#     shutil.move(image_list[i], ORIG_INPUT_DATASET + "\\evaluation\\" + image_list[i])
	# for i in x[e:]:
	#     shutil.move(image_list[i], ORIG_INPUT_DATASET + "\\training\\" + image_list[i])







	print("Building raw image folders ...")
	print("Building evaluation ...")
	for i in index[0]:
		if (len(i.split("_")) <3):
			continue
		image_name = i+"_"+name_base + ".jpg"
		move(image_name, "ForG\\evaluation\\" + image_name)


	print("Building training ...")
	for i in index[1]:
		if (len(i.split("_")) <3):
			continue
		image_name = i+"_"+name_base+".jpg"
		move(image_name, "ForG\\training\\" + image_name)



	build_dataset(TRAIN, TEST, VAL, ORIG_INPUT_DATASET, CLASSES, BASE_PATH)




	rmtree('ForG')


	extract_features(TRAIN, TEST, BASE_PATH, BASE_CSV_PATH, BATCH_SIZE, LE_PATH)
	# train(info = "Result of Right Mic with Feature Extraction - VGG\nRight Mic\nRevised each of the spectrogram\n\t1. with half of overlap.\n\t2. with half of overlap and half of window length\nTotal amount of images 30000")
	train("Result of Right Mic with Feature Extraction - VGG\nRandom choose 500 gap 500 foliage for all 10 datasets \nRevised each of the spectrogram\n\twindow length: longest, overlap: 0\nTotal amount of images 30000", BASE_CSV_PATH, TRAIN, TEST, MODEL_PATH, LE_PATH)

	print("Archiving file ...")
	make_archive(name_base, 'zip', BASE_PATH)

	print("Removing files ...")
	rmtree(BASE_PATH)
	os.remove(BASE+"\\"+"output"+"\\"+TRAIN+".csv")
	os.remove(BASE+"\\"+"output"+"\\"+TEST +".csv")

	e = time.time()
	print("Duration: " +  str(e-s))

