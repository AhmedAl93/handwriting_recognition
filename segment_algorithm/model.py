#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
#from matplotlib.patches import Rectangle
#from scipy.misc import imresize

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from imgprocess import *


# In[ ]:


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[ ]:



# from all 'data', get 'num'
# assume data is 'list'
def my_next_batch(num, data, labels): 
    #idx = np.arange(len(data)) 
    #np.random.shuffle(idx) 
    #idx = idx[:num] 
    idx = np.random.choice(len(data), num, replace=False)
    data_shuffle = data[idx] 
    labels_shuffle = labels[idx] 
    #labels_shuffle = np.asarray(labels_shuffle.reshape(len(labels_shuffle), 1)) 
    return data_shuffle, labels_shuffle


# In[ ]:


def LeNet_4_deep(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    conv1_w = tf.Variable(tf.truncated_normal(shape = [4,4,1,16],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'SAME') + conv1_b 
    conv1 = tf.nn.relu(conv1)
    pool_1 = tf.nn.avg_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #print(conv1, '\n', pool_1, '\n')
    
    conv2_w = tf.Variable(tf.truncated_normal(shape = [3,3,16,64], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'SAME') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool_2 = tf.nn.avg_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    #print(conv2, '\n', pool_2, '\n')
   
    conv3_w = tf.Variable(tf.truncated_normal(shape = [3,3,64,256], mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(256))
    conv3 = tf.nn.conv2d(pool_2, conv3_w, strides = [1,1,1,1], padding = 'SAME') + conv3_b
    conv3 = tf.nn.relu(conv3)
    pool_3 = tf.nn.avg_pool(conv3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    #print(conv3, '\n', pool_3, '\n')
    
    fc1 = flatten(pool_3)
    
    #print(fc1, '\n')

    fc1_w = tf.Variable(tf.truncated_normal(shape = (4096,512), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    
    #print(fc1, '\n')
    
    
    fc2_w = tf.Variable(tf.truncated_normal(shape = (512,75), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(75))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    #print(fc2, '\n')
    
    fc3_w = tf.Variable(tf.truncated_normal(shape = (75,11), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(11))
   
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    #print(logits, '\n')
    
    return logits


# In[ ]:


def construct_graph():
    # reset_graph()
    #EPOCHS = 1000 # 10000
    #BATCH_SIZE = 128 # 128

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 11) # 62 classes

    rate = 1e-3 # used 0.001
    logits = LeNet_4_deep(x)
    Y_proba = tf.nn.softmax(logits, name="Y_proba")
    # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    # init = tf.global_variables_initializer()


# In[ ]:


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[ ]:


def resize_norm_1(predict_img, target_h = 32, target_w = 32, showplot = False):
    X_predict = []
    for img in predict_img:
        img = img[np.newaxis, :, :, np.newaxis]
        new_img = tf.image.resize_area(img, [target_h, target_w]) # area > bilinear > bicubic ?_images
        #new_img = tf.image.resize_image_with_crop_or_pad(img, target_h, target_w) 
        new_img = new_img/255
        X_predict.append(new_img)
        
    X_predict = tf.concat(X_predict, axis=0)
    
    sess = tf.get_default_session()
    X_predict = sess.run(X_predict)
    
    #print(X_predict.shape)
    if showplot:
        fig, axes = plt.subplots(3, 4, figsize = (5, 5))

        for i in range(len(X_predict)):
            r_id, c_id = i//4, i%4
            axes[r_id, c_id].imshow(X_predict[i][:, :, 0], cmap = 'gray')
            
    return X_predict


# In[ ]:


def execute_graph():
    with tf.Session() as sess:
        saver.restore(sess, "./saved_model/lenet_4_deep")
        X_pred = resize_norm(predict_img_clean, 32, 32, showplot = True)  ##predict_img_clean or predict_img
        ####X_pred = resize_norm_1(predict_img, 32, 32, showplot=True)
        #X_pred = np.pad(X_pred, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        #logits_val = logits.eval(feed_dict={x: X_pred})

        Y_proba_val = Y_proba.eval(feed_dict={x: X_pred})
        y_pred = np.argmax(Y_proba_val, axis=1)


    #results = []    
    #for i in y_pred:
    #    results.append(resultmap[i])

    print('top candidate: ', y_pred)
    plt.show()
    
    return y_pred, Y_proba_val


# In[ ]:


def find_top_k_candidates(y_pred, Y_proba_val, k = 3):
    # find top k results

    results_top = []
    for i in range(len(Y_proba_val)):
        indx = np.argsort(Y_proba_val[i])
        indx_pick = indx[::-1][:k]
        results_top.append(indx_pick)

    if results_top[0][0] == 10:
        y_pred[0] = 1
    if results_top[-1][0] == 10:
        y_pred[-1] = 1

    print('top candidate: ', y_pred)
    print('top 3 candidates: ', results_top)
    
    return y_pred, results_top


# In[ ]:


def get_final_date(y_pred, final_box_margin):
    # get distance between characters

    charposition = []
    for box in final_box_margin:
        charposition.append((box[0]+box[2])/2)
    chardistance = []
    for i in range(len(charposition)-1):
        chardistance.append(charposition[i+1] - charposition[i])


    # group characters based on their distance    
    groupchars = []
    i = 0
    while i < len(y_pred) - 1:
        cur_group = [i]
        while i < len(y_pred) - 1 and y_pred[i+1] != 10 and chardistance[i] < 47: #45
            cur_group.append(i+1)
            i += 1
        groupchars.append(cur_group)
        if i < len(y_pred) - 1 and y_pred[i+1] != 10:
            i += 1
        else:
            i += 2

        
    # make sure that we only have 3 groups, like (date)/(month)/(year)        
    while len(groupchars) > 3:

        checksep = []
        for i in range(len(groupchars)-1):
            checkbox = [final_box_margin[groupchars[i][-1]], final_box_margin[groupchars[i+1][0]]]
            sepbox = [min(checkbox[0][2], checkbox[1][0]),
                      max(checkbox[0][1], checkbox[1][1]), 
                      max(checkbox[1][0], checkbox[0][2]),
                      min(checkbox[0][3], checkbox[1][3])]
            checksep.append([i, sepbox])

        sepboxes = [k[1] for k in checksep]
        sepimgs, _ = getcharimg(~imgtext, sepboxes, threshval = 175, showplot=False)

        sepratio = []
        for sepimg in sepimgs:
            curratio = np.sum(sepimg[i] != 0)/ ( (sepboxes[i][3] - sepboxes[i][1]) * (sepboxes[i][2] - sepboxes[i][0]) )
            sepratio.append(curratio)
        indx = np.argmin(sepratio)

        rmsep = checksep[indx][0]

        groupchars[rmsep] = groupchars[rmsep] + groupchars[rmsep+1]
        groupchars.pop(rmsep+1)

    
    # get final result based on char groups
    predictdate = []

    for group in groupchars:
        cur_group = ''.join([str(y_pred[k]) for k in group])
        predictdate.append(cur_group)


    predictdate = '-'.join([k for k in predictdate]) # use '-' or '/' to put digits together 

    print('\nrecognized date is: ', predictdate)

    print('\noriginal scanned doc is: ')
    visualizebox(final_box_margin, imgtext)
    plt.show()
    
    return predictdate


# In[ ]:





# In[ ]:





# In[ ]:




