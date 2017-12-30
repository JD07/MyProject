from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from tensorflow.python.platform import gfile
import re

FLAGS = None

def read_pairs(pairsPath):
    pairs = []
    with open(pairsPath, 'r') as f:
        for line in f.readlines():
            pair = line.strip().split()
            pairs.append(pair[0])
            pairs.append(pair[1])
    return pairs

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def readImage(imageList):
    lenc = len(imageList)
    img_list = [None] * lenc
    for i in range(lenc):
        img = misc.imread(imageList[i], mode='RGB')
        img = misc.imresize(img, (FLAGS.image_size, FLAGS.image_size), interp='bilinear')
        prewhitened = prewhiten(img)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images   

def evaluate(embeddings, actual_issame, bestThreshold):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]#从序号0开始取到末尾，步长为2（也就是取embeding偶数序号的值）
    embeddings2 = embeddings[1::2]#取奇数序号的值
    #使用相邻的embeding值计算欧几里得距离
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    #计算准确率
    _, _, accuracy = calculate_accuracy(bestThreshold, dist, actual_issame)

    return accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    '''
        计算真正例率，假正例率以及准确率
        参数：
            threshold:判断是否为正的门限值
            dist:embedding值组成的np.array
            actual_issame:真值组成的np.array
        返回:
            tpr:真正例率
            fpr:假正例率
            acc:识别准确率
    '''
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def pairsJudge(sameList, diffList):
    with tf.Graph().as_default():

        with tf.Session() as sess:
            #构建路径list和真值list
            path_list = sameList + diffList
            issame_list = [True for i in range(len(sameList)//2)] + [False for i in range(len(diffList)//2)]
            
            # 读取模型
            load_model(FLAGS.model)
    
            # 获取输入输出tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            embedding_size = embeddings.get_shape()[1]
            nrof_images = len(sameList) + len(diffList)
            nrof_batches = int(math.ceil(1.0*nrof_images / FLAGS.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))

            #前像传播
            for i in range(nrof_batches):
                start_index = i*FLAGS.batch_size
                end_index = min((i+1)*FLAGS.batch_size, nrof_images)
                paths_batch = path_list[start_index:end_index]
                images = readImage(paths_batch)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            #计算准确率
            accuracy = evaluate(emb_array, issame_list, FLAGS.bestThreshold)
            print ("verification accuracy: {:.4f}".format(accuracy))



def main(args):
    #读取相同pairs
    samePairsPath = os.path.join(FLAGS.pairsPath, 'same_pairs.txt')
    sameList = read_pairs(samePairsPath)
    #读取不同pairs
    diffPairsPath = os.path.join(FLAGS.pairsPath, 'diff_pairs.txt')
    diffList = read_pairs(diffPairsPath)   
    #执行网络
    pairsJudge(sameList, diffList)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str, 
                        help='The path of pretained model',
                        default='model')
    parser.add_argument('--pairsPath', 
                        type=str, 
                        help='The path of pairs to compare',
                        default='images_aligned_sample')
    parser.add_argument('--image_size', 
                        type=int,
                        help='Image size (height, width) in pixels.', 
                        default=160)
    parser.add_argument('--batch_size', 
                        type=int,
                        help='', 
                        default=32)
    parser.add_argument('--bestThreshold', 
                        type=int,
                        help='bestThreshold counted by another .py', 
                        default=1.0809)

    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)