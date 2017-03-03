# Load pickled data
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
from collections import Counter

training_file = 'lab 2 data/train.p'
testing_file = 'lab 2 data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']

classNames = {}
with open('signnames.csv', 'r') as csvfile:
    namereader = csv.reader(csvfile)
    next(namereader, None)
    for num, name in namereader:
        classNames[int(num)] = name

yName = [classNames[_y] for _y in  y_train]
# Plot histogram of classes
counts = Counter(yName)
df = pandas.DataFrame.from_dict(counts, orient='index')

df.plot(kind='bar', legend=False)
plt.title('Class counts')
plt.waitforbuttonpress()




