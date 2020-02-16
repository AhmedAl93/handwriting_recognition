#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
#from matplotlib.patches import Rectangle
#from scipy.misc import imresize
import copy
import cv2

from imgprocess import *
from model import *


# In[ ]:





# In[ ]:





# In[5]:


def load_img(imgpath, box = [1000, 2140, 1800, 2300]):
    img = cv2.imread(imgpath)
    x1, y1, x2, y2 = box
    img = img[y1:y2, x1:x2,  :]
    #plt.imshow(img)
    return img


# In[ ]:





# In[ ]:


## test

if __name__ == '__main__':
    imgpath = '../sample/test.jpg'
    img = load_img(imgpath)

    pre = preprocess(img)
    h_sum, v_sum = projection(pre)
    textbox = findtextbox(h_sum, v_sum)
    textbox_final = cleantextbox(img, textbox, tol = 5)
    # showtextbox(img, textbox_final) # if want to show seg

    imgtext, boxes, boxwidth = rough_seg(img, textbox_final)
    final_box_margin = fine_seg(boxes, boxwidth, imgtext)
    predict_img, original_shape = getcharimg(~imgtext, final_box_margin, threshval = 175, showplot = True)
    predict_img_clean = cleancharimg(predict_img, showplot = True)


    reset_graph()
    construct_graph()
    y_pred, Y_proba_val = execute_graph()
    y_pred, results_top = find_top_k_candidates(y_pred, Y_proba_val, k = 3)
    final_prediction = get_final_date(y_pred, final_box_margin) # most possible result


# In[ ]:





# In[ ]:




