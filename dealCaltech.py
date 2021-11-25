
file = open("data/caltech.train")
file2 = open("data/caltechtrain.train", 'a+')
while 1:
    line = file.readline()
    if not line:
        break
    list = line.split('/')
    basename = list[-1]
    line = 'Caltech/images/' + basename
    file2.write(line)

file.close()
file2.close()   
print('end!')