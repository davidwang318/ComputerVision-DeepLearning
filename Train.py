#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Lih-Narn Wang (ytcdavid@terpmail.umd.edu)
M.Eng in Robotics,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)
from Misc.MiscUtils import *
from Misc.DataUtils import *
import Misc.ImageUtils as iu

import tensorflow as tf
import cv2
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import CIFAR10Model
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True
    
def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, NumValidSamples, train):
    """
    Inputs: 
    BasePath - Path to CIFAR10 folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    ImageNum = 0
    if train:
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(NumValidSamples, len(DirNamesTrain)-1)
            
            RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.png'   
            ImageNum += 1

            I1 = np.float32(cv2.imread(RandImageName))
            # I1 = normalization(I1, axis=(1, 2))
            Label = convertToOneHot(TrainLabels[RandIdx], 10)
            I1Batch , LabelBatch= iu.preprocess1(I1, Label, I1Batch, LabelBatch)

    else:
        for i in range(NumValidSamples):
            ImageName = BasePath + os.sep + DirNamesTrain[i] + '.png'   
            I1 = np.float32(cv2.imread(ImageName))
            # I1 = normalization(I1, axis=(1, 2))
            Label = convertToOneHot(TrainLabels[i], 10)
            # Append All Images and Mask
            I1Batch.append(I1)
            LabelBatch.append(Label)
    return I1Batch, LabelBatch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)         

    

def TrainOperation(ImgPH, LabelPH, statePH, DirNamesTrain, TrainLabels, NumTrainSamples, NumValidSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to CIFAR10 folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath

    NumValidSamples - first() images of training set
    statePH is the state of the procedure (true for training, false for validation)
    """      
    # Predict output with forward pass
    learnRate = 0.0001
    accPlot, lossPlot, valPlot = [], [], []
    modelName = 'GitHub_Testing'
    prLogits, prSoftMax = CIFAR10Model(ImgPH, ImageSize, MiniBatchSize, statePH)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prLogits, labels=LabelPH))

    with tf.name_scope('Accuracy'):
        prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
        LabelDecoded = tf.argmax(LabelPH, axis=1)
        Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))
        
    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=learnRate).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.scalar('Accuracy', Acc)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    # Setup Saver
    Saver = tf.train.Saver()
    # Setup Validation set accuracy
    validAccBest = 0
    
    with tf.Session() as sess:       
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            StartEpoch = 0
            # Extract only numbers from the name
            # StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('\n\nNew model initialized....')
            #Saver.restore(sess, ModelPath)
            print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            accPerBatch, lossPerBatch = 0, 0

            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, NumValidSamples, True)
                FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch, statePH: True}
                _, accThisBatch, LossThisBatch, Summary = sess.run([Optimizer, Acc, loss, MergedSummaryOP], feed_dict=FeedDict)
                accPerBatch = accPerBatch + accThisBatch
                lossPerBatch = lossPerBatch + LossThisBatch

                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()
            # Generate validation set with batch size = 1
            # Two network will comsume all memories in GPU so I set the batch size to minimum
            validImg, validLabel = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize, NumValidSamples, False)
            validAcc = 0
            for i in range(len(validImg)):
            	validDict = {ImgPH: [validImg[i]], LabelPH: [validLabel[i]], statePH: False}
            	tmp = sess.run(Acc, feed_dict=validDict)
            	validAcc = validAcc + tmp
            validAcc = validAcc/len(validImg)
            # Print the information of training process
            print('===============')
            print('acc', accPerBatch/NumIterationsPerEpoch)
            print('loss', lossPerBatch/NumIterationsPerEpoch)
            print('valid_acc', validAcc)
            print('===============')
            # Save best validation model
            if validAcc > validAccBest:
	            validAccBest = validAcc
	            print('New Best')
	            SaveName = CheckPointPath + modelName + 'Bestmodel.ckpt'
	            Saver.save(sess, save_path=SaveName)
	            #print('\n' + SaveName + ' Model Saved...')
            # Plot accuracy for each epoch
            accPlot.append(accPerBatch/NumIterationsPerEpoch)
            lossPlot.append(lossPerBatch/NumIterationsPerEpoch)
            valPlot.append(validAcc)
        Plot(accPlot, modelName, 'TraingAccuracy')
        Plot(lossPlot, modelName, 'TrainLoss')
        Plot(valPlot, modelName, 'ValidationAccuracy')

	            

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../CIFAR10', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/CIFAR10')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=100, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath


    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)

    # Take first 5000 images as validation set
    validSize = 5000
    NumTrainSamples = NumTrainSamples - validSize

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(None, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.placeholder(tf.float32, shape=(None, NumClasses)) # OneHOT labels
    statePH = tf.placeholder(tf.bool, shape=())
    
    TrainOperation(ImgPH, LabelPH, statePH, DirNamesTrain, TrainLabels, NumTrainSamples, validSize, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath)
        
    
if __name__ == '__main__':
    main()
 

