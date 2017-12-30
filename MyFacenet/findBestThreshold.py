from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import random
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from scipy import misc

def listDivider(fileList, ratio = 0.5):
    '''
        将指定list随机分成两部分
        参数:
            fileList:源列表
            ratio:分割比例
        返回：
            resultList1,resultList2:分割后的列表
    '''
    totalNum = len(fileList)
    threshold = int(totalNum*ratio)
    index = list(range(totalNum))
    random.shuffle(index)
    resultList1 = [fileList[i] for i in index[0:threshold]]
    resultList2 = [fileList[i] for i in index[threshold: ]]
    return resultList1, resultList2

def getImagePath(path):
    folderList = os.listdir(path)
    sampleList,_ = listDivider(folderList, ratio = 0.5)
    sameList, diffList = listDivider(sampleList, ratio = 0.33)
    images = []
    issame = []
    #抽取相同类别样本
    for folder in sameList:
        imagePath = os.path.join(path, folder)
        imageList = os.listdir(imagePath)
        numImage = len(imageList)
        if numImage < 10:
            continue
        index = list(range(numImage))
        random.shuffle(index)
        images.append(os.path.join(imagePath, imageList[0]))
        images.append(os.path.join(imagePath, imageList[1]))
        issame.append(True)
    #抽取不同类别样本
    lenc = len(diffList)//2
    for i in range(lenc):
        imagePath = os.path.join(path, diffList[2*i])
        imageList = os.listdir(imagePath)
        random.shuffle(imageList)
        images.append(os.path.join(imagePath, imageList[0]))
        imagePath = os.path.join(path, diffList[2*i+1])
        imageList = os.listdir(imagePath)
        random.shuffle(imageList)
        images.append(os.path.join(imagePath, imageList[0]))   
        issame.append(False)
    return images,issame

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]#从序号0开始取到末尾，步长为2（也就是取embeding偶数序号的值）
    embeddings2 = embeddings[1::2]#取奇数序号的值
    #计算TPR，FPR以及accuracy
    thresholds = np.arange(0, 2.5, 0.001)
    tpr, fpr, accuracy, bestThreshold = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
                                                np.asarray(actual_issame), nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, bestThreshold

def readImage(imageList):
    lenc = len(imageList)
    img_list = [None] * lenc
    for i in range(lenc):
        img = misc.imread(imageList[i], mode='RGB')
        img = misc.imresize(img, (FLAGS.image_size, FLAGS.image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(img)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images   

def findBestThreshold(path_list, issame_list):
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(FLAGS.model)            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")            

            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')
            nrof_images = len(path_list)
            nrof_batches = int(math.ceil(1.0*nrof_images / FLAGS.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches):
                start_index = i*FLAGS.batch_size
                end_index = min((i+1)*FLAGS.batch_size, nrof_images)
                paths_batch = path_list[start_index:end_index]
                images = readImage(paths_batch)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
        
            tpr, fpr, accuracy, bestThreshold = evaluate(emb_array, 
                                         issame_list, nrof_folds=10)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Best Threshold', np.mean(bestThreshold))

def main(args):
    path_list, issame_list = getImagePath(FLAGS.imagePath)
    findBestThreshold(path_list, issame_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str, 
                        help='The path of pretained model',
                        default='model')
    parser.add_argument('--imagePath', 
                        type=str, 
                        help='The path of images',
                        default='data/webface')
    parser.add_argument('--image_size', 
                        type=int,
                        help='Image size (height, width) in pixels.', 
                        default=160)
    parser.add_argument('--batch_size', 
                        type=int,
                        help='', 
                        default=32)                        

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)