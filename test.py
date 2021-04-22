import numpy as np

index = np.load("e_t_index.npy", allow_pickle=True)

new_index = []

for l in index:
	temp = []
	for i in l:
		first = i.split("_")[0]
		if(first != "0" and first != "1"):
			continue
		temp.append(i)
	new_index.append(temp)
np.save("e_t_index.npy", new_index)