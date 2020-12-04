def read_dataset1 ():
	f = open("project3_dataset1.txt", "r")
	Lines = f.readlines()
	data = []
	for line in Lines:
		
		line = line.replace('\t',' ')
		line = line.replace('\n','')
		ls = list(line.split(" "))
		ls = [float(i) for i in ls]
		data.append(ls)
	X = [ls[:-1] for ls in data]
	y = [ls[-1] for ls in data]
	return (X, y)

# print(len(data))		#length of observation
# print(len(data[0]))		#length of features

def read_dataset2 ():
	f = open("project3_dataset2.txt", "r")
	Lines = f.readlines()
	data = []
	for line in Lines:
		
		line = line.replace('\t',' ')
		line = line.replace('\n','')
		ls = list(line.split(" "))
		ls = [float(i) for i in ls]
		data.append(ls)
	X = [ls[:-1] for ls in data]
	y = [ls[-1] for ls in data]
	return (X, y)