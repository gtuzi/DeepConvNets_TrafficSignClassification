
# Copyright (c) 2017, Gerti Tuzi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Gerti Tuzi nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tensorflow as tf

def GTInception(x, keep_prob):
    mu = 0
    sigma = 0.01
    mult = 10

    # 32*32*3 to 28*28*6*mult
    Wc1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 6 * mult], mean=mu, stddev=sigma))
    bc1 = tf.Variable(tf.zeros([6 * mult]))
    c1 = conv2d(x=x, W=Wc1, b=bc1, strides=[1, 1, 1, 1], padding='VALID')

    # --------------------------- #
    # 28*28*6*mult to 14*14*6*mult
    i1k1 = 3 * mult
    i1k3 = 1 * mult
    i1k5 = 1 * mult
    i1kp = 1 * mult
    c1i, Wc1i = InceptionModule(x=c1, xdepth=6 * mult, k1=i1k1, k3=i1k3, k5=i1k5, kp=i1kp)
    s2 = maxpool2d(c1i, k=[1, 2, 2, 1], padding='SAME')

    # --------------------------- #
    # 14 * 14 * 6*mult to 14*14*10*mult
    i3k1 = 3 * mult
    i3k3 = 3 * mult
    i3k5 = 2 * mult
    i3kp = 2 * mult
    c3i, Wc3i = InceptionModule(x=s2, xdepth=6 * mult, k1=i3k1, k3=i3k3, k5=i3k5, kp=i3kp)

    # --------------------------- #
    # 14*14*10*mult to 14*14*16*mult
    i4k1 = 7 * mult
    i4k3 = 5 * mult
    i4k5 = 2 * mult
    i4kp = 2 * mult
    c4i, Wc4i = InceptionModule(x=c3i, xdepth=10 * mult, k1=i4k1, k3=i4k3, k5=i4k5, kp=i4kp)

    # --------------------------- #
    # 14*14*16*mult to 7*7*16*mult
    i5k1 = 5 * mult
    i5k3 = 6 * mult
    i5k5 = 3 * mult
    i5kp = 2 * mult
    c5i, Wc5i = InceptionModule(x=c4i, xdepth=16 * mult, k1=i5k1, k3=i5k3, k5=i5k5, kp=i5kp)
    s5 = maxpool2d(c5i, k=[1, 2, 2, 1], padding='SAME')

    # --------------------------- #
    # 7*7*16*mult to 7*7*32*mult
    i6k1 = 10 * mult
    i6k3 = 14 * mult
    i6k5 = 5 * mult
    i6kp = 3 * mult
    c6i, Wc6i = InceptionModule(x=s5, xdepth=16 * mult, k1=i6k1, k3=i6k3, k5=i6k5, kp=i6kp)

    # --------------------------- #
    # 7*7*32*mult to 7*7*64*mult
    i7k1 = 17 * mult
    i7k3 = 32 * mult
    i7k5 = 10 * mult
    i7kp = 5 * mult
    c7i, Wc7i = InceptionModule(x=c6i, xdepth=32 * mult, k1=i7k1, k3=i7k3, k5=i7k5, kp=i7kp)

    # --------------------------- #
    # 7*7*64*mult to 64*mult
    s8 = maxpool2d(c7i, k=[1, 7, 7, 1], padding='SAME')

    # --------------------------- #
    # Flat input w/ dropout: 64*mult
    fc9 = tf.contrib.layers.flatten(s8)
    fc9 = tf.nn.dropout(fc9, keep_prob)

    # ----------------------
    # Fully connected 64*mult --> 1024
    Wfc10 = tf.Variable(tf.truncated_normal(shape=[64 * mult, 1024], mean=mu, stddev=sigma))
    bfc10 = tf.Variable(tf.zeros([1024]))
    fc10 = tf.nn.bias_add(tf.matmul(fc9, Wfc10), bfc10)
    fc10 = tf.nn.relu(fc10)
    fc10 = tf.nn.dropout(fc10, keep_prob)

    # ----------------------
    # Output layer (logits): 1024 --> 43
    Wfc11 = tf.Variable(tf.truncated_normal(shape=[1024, 43], mean=mu, stddev=sigma))
    bfc11 = tf.Variable(tf.zeros([43]))
    fc11 = tf.nn.bias_add(tf.matmul(fc10, Wfc11), bfc11)

    out = fc11

    Wout = tf.concat(values=[tf.reshape(Wc1, shape=(1, -1)),
                             tf.reshape(Wc1i, shape=(1, -1)),
                             tf.reshape(Wc3i, shape=(1, -1)),
                             tf.reshape(Wc4i, shape=(1, -1)),
                             tf.reshape(Wc5i, shape=(1, -1)),
                             tf.reshape(Wc6i, shape=(1, -1)),
                             tf.reshape(Wc7i, shape=(1, -1)),
                             tf.reshape(Wfc10, shape=(1, -1)),
                             tf.reshape(Wfc11, shape=(1, -1))],
                     concat_dim=1)

    return out, Wout


def InceptionModule(x, xdepth, k1, k3, k5, kp):
    # Input: x - input tensor from previous layer
    #        nchannels - depth of previous layer
    #        k1/k3/k5 - depth of 1x1, 3x3, 5x5 convolutions
    #
    # Output: Concatenate along depth (i.e. channels) nchannels = k1 + k2 + k3 + kp
    #         Spatial dimensions are kept the same by zero-padding (i.e. using SAME)
    mu = 0
    sigma = 0.1

    # 1x1 Convolution
    W1 = tf.Variable(tf.truncated_normal([1, 1, xdepth, k1], mean=mu, stddev=sigma))
    b1 = tf.Variable(tf.zeros(k1))
    C1 = tf.nn.bias_add(tf.nn.conv2d(input=x, filter=W1, strides=[1, 1, 1, 1], padding='SAME'), b1)
    C1 = tf.nn.relu(C1)

    # 3x3 Convolution
    W3 = tf.Variable(tf.truncated_normal([3, 3, k1, k3], mean=mu, stddev=sigma))
    b3 = tf.Variable(tf.zeros(k3))
    C3 = tf.nn.bias_add(tf.nn.conv2d(input=C1, filter=W3, strides=[1, 1, 1, 1], padding='SAME'), b3)
    C3 = tf.nn.relu(C3)

    # 5x5 Convolution
    W5 = tf.Variable(tf.truncated_normal([5, 5, k1, k5], mean=mu, stddev=sigma))
    b5 = tf.Variable(tf.zeros(k5))
    C5 = tf.nn.bias_add(tf.nn.conv2d(input=C1, filter=W5, strides=[1, 1, 1, 1], padding='SAME'), b5)
    C5 = tf.nn.relu(C5)

    # 3x3 maxpooling followed by a 1x1 convolution
    mpooling = tf.nn.max_pool(value=x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    Wp = tf.Variable(tf.truncated_normal([1, 1, xdepth, kp], mean=mu, stddev=sigma))
    bp = tf.Variable(tf.zeros(kp))
    Cp = tf.nn.bias_add(tf.nn.conv2d(input=mpooling, filter=Wp, strides=[1, 1, 1, 1], padding='SAME'), bp)
    Cp = tf.nn.relu(Cp)

    # Concatenate along depth
    out = tf.concat(values=[C1, C3, C5, Cp], concat_dim=3)

    # Return weights (to be used for post processing. ex. regularization)
    Wout = tf.concat(values=[tf.reshape(W1, shape=(1, -1)),
                             tf.reshape(W3, shape=(1, -1)),
                             tf.reshape(W5, shape=(1, -1)),
                             tf.reshape(Wp, shape=(1, -1))],
                     concat_dim=1)

    return out, Wout


def conv2d(x, W, b, strides=[1, 1, 1, 1], padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=strides, padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def avgpool2d(x, k=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.avg_pool(value=x, ksize=k, strides=k, padding=padding)


def maxpool2d(x, k=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(x, ksize=k, strides=k, padding=padding)

