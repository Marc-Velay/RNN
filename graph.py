from csv import reader
from matplotlib import pyplot

with open('dataSet.csv', 'r') as f:
    data = list(reader(f))
    temp = [i[1] for i in data]
    temp2 = [i[2] for i in data]
    pyplot.plot(range(len(temp)), temp)
    pyplot.plot(range(len(temp2)), temp2)
    pyplot.show()