import numpy as np

def read_dataset (path):
	f = open(path, "r")
	Lines = f.readlines()
	data = []
	for line in Lines:
		
		line = line.replace('\t',' ')
		line = line.replace('\n','')
		ls = list(line.split(" "))
		# print(ls)
		for i in range(len(ls)):
			if ls[i] == 'Present':
				ls[i] = 1
			elif ls[i] == 'Absent':
				ls[i] = 0

		ls = [float(i) for i in ls]
		data.append(ls)
	X = [ls[:-1] for ls in data]
	y = [ls[-1] for ls in data]
	return (np.array(X), np.array(y))

# print(len(data))		#length of observation
# print(len(data[0]))		#length of features
