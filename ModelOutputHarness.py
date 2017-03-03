# Load pickled data

import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
import matplotlib.image as mpimg
from GTInception import GTInception


classNames = {}
with open('signnames.csv', 'r') as csvfile:
    namereader = csv.reader(csvfile)
    next(namereader, None)
    for num, name in namereader:
        classNames[int(num)] = name


imgfolder = 'web_images/'
imageNames = {0 : '30kmh.jpg', 1:'danger.jpg',
              2:'dogpoop.jpg', 3:'dontpass.jpg',
              4:'turn.jpg', 5:'danger1.jpg'}



# Set up the tf variables
tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
global_step = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)


logits, Wout = GTInception(x, keep_prob)
net_op = tf.nn.softmax(logits, dim=-1)
pred_op = tf.nn.top_k(net_op, k = 5, name='Top5Select')


results = {}

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, save_path='./models/GTInception')

    for key, imgname in imageNames.items():
        imgfile = imgfolder + imgname
        image = mpimg.imread(imgfile)
        img_exp = np.expand_dims(image, axis=0)
        values, classes = sess.run(pred_op, feed_dict={x: img_exp.astype(dtype='float32'), keep_prob:1.0})
        classes = [classNames[int(c)] for c in np.nditer(classes)]
        results[key] = (values, classes, image)


for key, val in results.items():
    values, classes, image = val[0], val[1], val[2]

    df = pandas.DataFrame.from_items(zip(classes, values.tolist()))
    figure = plt.figure()
    ax = figure.add_subplot(210 + 1 + 0)
    ax.imshow(image.squeeze())
    ax = figure.add_subplot(210 + 1 + 1)
    ax = df.plot(kind='barh', legend=False, ax = ax)
    ax.set_yticklabels(classes, rotation=0)
    plt.title('Softmax output')
    plt.waitforbuttonpress()




