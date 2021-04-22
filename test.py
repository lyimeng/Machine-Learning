import numpy as np
import os




if __name__ == '__main__':
	dirlist = os.listdir("image_as_array_02_23_2021")
	count = 0
	count2 = 0
	for s in dirlist:
		s = "4_b"
		path = "image_as_array_02_23_2021\\%s\\"%s
		image_total = np.load(path + "predict_result_%s.npy"%s)
		for i in image_total:
			count += 1
			if (i == "2"):
				print(i)
				count2 += 1

		print("---------------------------------------------------------------------")
		print(s + " is done")
		print("---------------------------------------------------------------------")
	print(str(count2) + "out of" + str(count))