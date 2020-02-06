"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CIFAR10Model(Img, ImageSize, MiniBatchSize, state):

    #return naiveNet(Img, ImageSize, MiniBatchSize, state)
    return resNet12(Img, ImageSize, MiniBatchSize)
    #return denseNet(Img, ImageSize, MiniBatchSize, True)
    #return resNext(Img, ImageSize, MiniBatchSize)

# My first Neural Network
def naiveNet(Img, ImageSize, MiniBatchSize, state):
    # first layer
    conv1 = tf.layers.conv2d(inputs=Img, name='layer_conv1', padding='same', filters=64, kernel_size=5, activation=None)
    conv1 = tf.layers.batch_normalization(conv1, name='batch_norm1')
    conv1 = tf.nn.relu(conv1, name='relu_1')
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
    # Second layer
    conv2 = tf.layers.conv2d(inputs=conv1, name='layer_conv2', padding='same', filters=54, kernel_size=5, activation=None)
    conv2 = tf.layers.batch_normalization(conv2, name='batch_norm2')
    conv2 = tf.nn.relu(conv2, name='relu_2')
    conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)
    # Flatten
    flat1 = tf.contrib.layers.flatten(conv2)
    # Fully connected layers: 1, 2
    fc1 = tf.layers.dense(inputs=flat1, name='fc_1', units=256, activation=tf.nn.relu)
    fc1 = tf.layers.dropout(inputs=fc1, rate=0.5, training=state)
    fc2 = tf.layers.dense(inputs=fc1, name='fc_2', units=128, activation=tf.nn.relu)
    fc2 = tf.layers.dropout(inputs=fc2, rate=0.5, training=state)
    y = tf.layers.dense(inputs=fc2, name='softmax', units=10, activation=None)

    prLogits = y
    prSoftMax = tf.nn.softmax(y)

    return prLogits, prSoftMax
"""
resNet implementation
1 conv2d -> 5 idBlock -> global average -> dense layer for output
"""
def idBlock(x, size, blockName):
    # get dimension of the input data
    xShortcut = x
    dim = x.shape[3]
    size1, size2, size3 = size
    # First 
    conv1 = tf.layers.conv2d(inputs=x, name=blockName+'_conv_1', padding='same', filters=dim, kernel_size=size1, activation=None)
    conv1 = tf.layers.batch_normalization(conv1, name=blockName+'_batch_norm1')
    conv1 = tf.nn.relu(conv1, name=blockName+'_relu_1')
    # Second
    conv2 = tf.layers.conv2d(inputs=conv1, name=blockName+'_conv_2', padding='same', filters=dim, kernel_size=size2, activation=None)
    conv2 = tf.layers.batch_normalization(conv2, name=blockName+'_batch_norm2')
    conv2 = tf.nn.relu(conv2, name=blockName+'_relu_2')
    # Third
    conv3 = tf.layers.conv2d(inputs=conv2, name=blockName+'_conv_3', padding='same', filters=dim, kernel_size=size3, activation=None)
    conv3 = tf.layers.batch_normalization(conv3, name=blockName+'_batch_norm3')

    convAdd = tf.add(conv3, xShortcut)
    y = tf.nn.relu(convAdd, name=blockName+'_relu_3')

    return y

def convBlock(x, size, blockName):
    # get dimension of the input data
    dim = x.shape[3]
    size1, size2, size3, size4 = size
    # First 
    conv1 = tf.layers.conv2d(inputs=x, name=blockName+'_conv_1', padding='same', filters=dim, kernel_size=size1, activation=None)
    conv1 = tf.layers.batch_normalization(conv1, name=blockName+'_batch_norm1')
    conv1 = tf.nn.relu(conv1, name=blockName+'_relu_1')
    # Second
    conv2 = tf.layers.conv2d(inputs=conv1, name=blockName+'_conv_2', padding='same', filters=dim, kernel_size=size2, activation=None)
    conv2 = tf.layers.batch_normalization(conv2, name=blockName+'_batch_norm2')
    conv2 = tf.nn.relu(conv2, name=blockName+'_relu_2')
    # Third
    conv3 = tf.layers.conv2d(inputs=conv2, name=blockName+'_conv_3', padding='same', filters=dim, kernel_size=size3, activation=None)
    conv3 = tf.layers.batch_normalization(conv3, name=blockName+'_batch_norm3')
    # Shortcut
    xShortcut = tf.layers.conv2d(inputs=x, name=blockName+'_conv_shortcut', padding='same', filters=dim, kernel_size=size4, activation=None)
    xShortcut = tf.layers.batch_normalization(xShortcut, name=blockName+'_batch_norm')

    convAdd = tf.add(conv3, xShortcut)
    y = tf.nn.relu(convAdd, name=blockName+'_relu_3')

    return y

def resNet12(Img, ImageSize, MiniBatchSize):
    idBlockNum, convBlockNum = 2, 1
    # first layer
    conv1 = tf.layers.conv2d(inputs=Img, name='layer_conv1', padding='same', filters=64, kernel_size=5, activation=None)
    conv1 = tf.layers.batch_normalization(conv1, name='batch_norm1')
    conv1 = tf.nn.relu(conv1, name='relu_1')
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    for i in range(idBlockNum):
        conv2 = idBlock(conv1, [3,3,3], 'idBlock'+str(i))
    for i in range(convBlockNum):
        conv2 = convBlock(conv2, [3,3,3,3], 'convBlock'+str(i))

    conv2 = tf.reduce_mean(conv2, axis=[1,2])
    flat1 = tf.contrib.layers.flatten(conv2)
    y = tf.layers.dense(inputs=flat1, name='softmax', units=10, activation=None)

    prLogits = y
    prSoftMax = tf.nn.softmax(y)

    return prLogits, prSoftMax

"""
DenseNet implementation
helper function: bottleNeck(), transition(),denseBlock()
"""

def bottleNeck(x, blockName, filterSize, dropRate, state):
    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_batch_norm1')
    x = tf.nn.relu(x, name=blockName+'_relu_1')
    x = tf.layers.conv2d(inputs=x, name=blockName+'_layer_conv1', padding='same', filters=filterSize, kernel_size=1, activation=None)
    x = tf.layers.dropout(inputs=x, rate=dropRate, training=state)

    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_batch_norm2')
    x = tf.nn.relu(x, name=blockName+'_relu_2')
    x = tf.layers.conv2d(inputs=x, name=blockName+'_layer_conv2', padding='same', filters=filterSize, kernel_size=3, activation=None)
    x = tf.layers.dropout(inputs=x, rate=dropRate, training=state)
    return x

def transition(x, blockName,dropRate, state):
    filterSize = x.shape[3]
    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_batch_norm1')
    x = tf.nn.relu(x, name=blockName+'_relu_1')
    x = tf.layers.conv2d(inputs=x, name=blockName+'_layer_conv1', padding='same', filters=filterSize, kernel_size=1, activation=None)
    x = tf.layers.dropout(inputs=x, rate=dropRate, training=state)
    x = tf.layers.average_pooling2d(inputs=x, pool_size=2, strides=2)
    return x

def denseBlock(x, blockName, blockNum, growthRate,dropRate, state):
    filterSize = growthRate
    xConcat = x
    for i in range(blockNum):
        x = bottleNeck(xConcat, blockName+'_bottle'+str(i), filterSize, dropRate, state)
        xConcat = tf.concat([x, xConcat], 3)
    return xConcat

def denseNet(Img, ImageSize, MiniBatchSize, state):
    growthRate = 8
    x = tf.layers.conv2d(inputs=Img, name='layer_conv1', padding='same', filters=32, kernel_size=3, activation=None)
    for i in range(3):
        x = denseBlock(x, 'denBlock'+str(i), 5, growthRate, 0.5, state)
        print(x)
        x = transition(x, 'denBlock'+str(i), 0.5, state)
        print(x)
    x = tf.layers.batch_normalization(inputs=x, name='Dense_batch_norm')
    x = tf.nn.relu(x, name='Dense_relu')
    d1 = tf.reduce_mean(x, axis=[1,2])
    flat1 = tf.contrib.layers.flatten(d1)
    y = tf.layers.dense(inputs=flat1, name='softmax', units=10, activation=None)

    prLogits = y
    prSoftMax = tf.nn.softmax(y)
    return prLogits, prSoftMax

"""
ResNext Implementation
helper function:
    split(): for spliting the data and perform the transform for features.
    transform: to perform classic resnet transform in split().
    transmerge: to perform linear transform for merging with original input and output.
"""

def split(x, blockName, resDim,cardinality):
    flag = False
    xMerge = np.array([])
    for i in range(cardinality):
        xSplit = transform(x, blockName+str(i), resDim)
        xMerge = tf.concat([xMerge, xSplit], 3) if flag else xSplit
        flag = True
    return xMerge

def transform(x, blockName, dim):
    x = tf.layers.conv2d(inputs=x, name=blockName+'_transform_conv1', padding='same', filters=dim, kernel_size=1, activation=None)
    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_transform_bnorm1')
    x = tf.nn.relu(x, name=blockName+'_transform_relu1')

    x = tf.layers.conv2d(inputs=x, name=blockName+'_transform_conv2', padding='same', filters=dim, kernel_size=3, activation=None)
    return x

def transmerge(x, blockName, outputDim):
    x = tf.layers.conv2d(inputs=x, name=blockName+'_transmerge_conv1', padding='same', filters=outputDim, kernel_size=1, activation=None)
    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_transmerge_bnorm1')
    return x

def rNextBlock(xInput, blockName, blockNum, resDim, cardinality):
    outputDim = xInput.shape[3]
    for i in range(blockNum):
        x = split(xInput, blockName+str(i), resDim, cardinality)
        x = transmerge(x, blockName+str(i), outputDim)
    print(xInput, x)
    xOutput = tf.nn.relu(x+xInput, name=blockName+'relu')
    return xOutput

def resNext(x, ImageSize, MiniBatchSize):
    blockName = 'resNext'
    kSize = 64

    x = tf.layers.conv2d(inputs=x, name=blockName+'_conv1', padding='same', filters=kSize, kernel_size=3, activation=None)
    x = tf.layers.batch_normalization(inputs=x, name=blockName+'_bnorm1')
    x = tf.nn.relu(x, name=blockName+'_relu1')
    x = tf.layers.max_pooling2d(inputs=x, pool_size=2, strides=2)
    x = rNextBlock(x, 'NextBlock', 4, x.shape[3]/8, 8)

    x = tf.reduce_mean(x, axis=[1,2])
    flat1 = tf.contrib.layers.flatten(x)
    y = tf.layers.dense(inputs=flat1, name='softmax', units=10, activation=None)

    prLogits = y
    prSoftMax = tf.nn.softmax(y)
    
    return prLogits, prSoftMax




