#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from scipy.misc import imresize
import copy
import cv2


# In[ ]:





# In[ ]:


def preprocess(img, dilate_iter = 1, erode_iter = 10):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # smooth the image to avoid noises
    gray = cv2.medianBlur(gray,5)

    # Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst
    #src – Source 8-bit single-channel image.
    #dst – Destination image of the same size and the same type as src .
    #maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    #adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
    #thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    #blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.


    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = dilate_iter)
    thresh = cv2.erode(thresh,None,iterations = erode_iter)

    pre = ~thresh
    
    return pre


# In[ ]:


def projection(pre, showplot = True):
    
    # pre = preprocess(img)
    
    h_sum = np.sum(pre, axis=1)
    h_sum = h_sum/h_sum.max()
    
    v_sum = np.sum(pre, axis=0)
    v_sum = v_sum/v_sum.max()
 
    if showplot: 
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(h_sum, range(h_sum.shape[0]))
        ax[0].invert_yaxis()
        ax[1].plot(range(v_sum.shape[0]), v_sum)
    
    return h_sum, v_sum


# In[ ]:


# a function to group close lines

def grouplines(cols, min_sep):
    grouped = []
    i = 0
    while i < len(cols):
        checked = [i]
        cur_group = [cols[i]]
        j = i+1
        while j < len(cols):
            if cols[j] - cols[i] <= min_sep:
                cur_group.append(cols[j])
                checked.append(j)
                i = j
                j = j+1
            else:
                j +=1
        grouped.append(cur_group)
        i = checked[-1] + 1

    cols_clean = []
    for i in range(len(grouped)):
        #col = np.mean(grouped[i])
        col_left = np.min(grouped[i])
        col_right = np.max(grouped[i])
        cols_clean.append([int(col_left), int(col_right)])
        
    return grouped, cols_clean


# In[ ]:


def findrows(h_sum, row_thre = 0.1, row_sep = 2, row_min_width = 50):
    row = []
    for i in range(len(h_sum)):
        if h_sum[i] > row_thre:
            row.append(i)
    grouped_row, row_clean = grouplines(row, min_sep = row_sep)
    
    row_clean_1 = []
    for row_ele in row_clean:
        if row_ele[1] - row_ele[0] > row_min_width:
            row_clean_1.append(row_ele)

    return row_clean_1


# In[ ]:


def findcols(v_sum, col_thre = 0.5, col_sep = 10, start_position = 100, col_min_width = 150):
    col = []
    for i in range(len(v_sum)):
        if v_sum[i] > col_thre:
            col.append(i)
    grouped_col, col_clean = grouplines(col, min_sep = col_sep)
    col_clean = [k for k in col_clean if k[0] > start_position]
    
    col_clean_1 = []
    for col_ele in col_clean:
        if col_ele[1] - col_ele[0] > col_min_width:
            col_clean_1.append(col_ele)

    return col_clean_1


# In[ ]:


def findtextbox(h_sum, v_sum, row_thre = 0.1, row_sep = 2, row_min_width = 50, 
               col_thre = 0.5, col_sep = None, start_position = 100, col_min_width = 200): 
    
    row = findrows(h_sum, row_thre, row_sep, row_min_width)
    row = row[0]
    
    if col_sep is None:
        allcols = []
        for col_sep in [10, 15, 20, 25]:
            col = findcols(v_sum, col_thre, col_sep, start_position, col_min_width)
            if col != []:
                allcols.append(col[0])
        
        col_left = []
        col_right = []
        for col in allcols:
            col_left.append(col[0])
            col_right.append(col[1])
        col = [min(col_left), max(col_right)]
    else:
        col = findcols(v_sum, col_thre, col_sep, start_position, col_min_width)
    
    textbox = [col[0], row[0], col[1], row[1]]
    
    return textbox   
    


# In[ ]:


def cleantextbox(img, textbox, tol = 5):
    # to get final textbox, add a few pixel for margin
    h, w = img.shape[0], img.shape[1]

    x1 = textbox[0] - tol if (textbox[0] - tol >= 0) else textbox[0]
    y1 = textbox[1] - tol if (textbox[1] - tol >= 0) else textbox[1]
    x2 = textbox[2] + tol if (textbox[2] + tol <= w) else textbox[2]
    y2 = textbox[3] + tol if (textbox[3] + tol <= h) else textbox[3]

    textbox_final = [x1, y1, x2, y2]
    
    return textbox_final


# In[ ]:


def showtextbox(img, textbox):
    t = img.copy()
    pt1 = (textbox[0], textbox[1])
    pt2 = (textbox[2], textbox[3])
    cv2.rectangle(t, pt1, pt2, (0, 255, 0), thickness = 2)
    plt.figure(figsize = (10, 10))    
    plt.imshow(t)


# In[ ]:


def check_intersect(va, vb):
    x1_a, y1_a, x2_a, y2_a = va
    x1_b, y1_b, x2_b, y2_b = vb
    
    x = max(x1_a, x1_b)
    y = max(y1_a, y1_b)
    w = min(x2_a, x2_b) - x
    h = min(y2_a, y2_b) - y
    if w < 0 or h < 0:
        return False
    return w*h


# In[ ]:


def find_IOU(va, vb):
    x1_a, y1_a, x2_a, y2_a = va
    x1_b, y1_b, x2_b, y2_b = vb
    intersect_area = check_intersect(va, vb)
    
    if intersect_area:
        va_area = (x2_a - x1_a)*(y2_a - y1_a)
        vb_area = (x2_b - x1_b)*(y2_b - y1_b)
        union_area = va_area + vb_area - intersect_area
        
        iou = intersect_area/union_area
        return iou
    else: # no intersect 
        return 0


# In[ ]:


# search white & black bacground to find chars

def findchars(textpre, w_min = 5, w_max = 50, h_min = 30, h_max = None):
    
    if h_max is None:
        h_max = 0.8*textpre.shape[0]
        
    contours1, _ = cv2.findContours(textpre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(~textpre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = contours1 + contours2
    
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w_min<=w<=w_max and h_min<=h<=h_max:
            boxes.append([x, y, x+w, y+h])
   
    return boxes


# In[ ]:


# for clean boxes based on IOU

def cleaniou(boxes, iou_thre = 0.5):
    
    boxes.sort(key = lambda s: (s[0], s[2]))
    
    for i in range(len(boxes) - 1):
        for j in range(i+1, len(boxes)):
            iou = find_IOU(boxes[i], boxes[j])
            if iou >= iou_thre:
                if (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) >=                     (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]):
                    boxes[j] = boxes[i]
                else:
                    boxes[i] = boxes[j]
                    
    boxes_clean = []
    for box in boxes:
        if box not in boxes_clean:
            boxes_clean.append(box)
    boxes = boxes_clean
    
    return boxes_clean


# In[ ]:


# deal with large overlapping boxes

def cleanoverlap_part(sbox, lbox, box_thre = 10):
    
    add_boxes = []
    
    sboxwidth = sbox[2] - sbox[0]
    left_r = sbox[0] - lbox[0]
    right_r = lbox[2] - sbox[2]

    # two possible boxes
    new_b_1 = [lbox[0], min(sbox[1], lbox[1]), sbox[0], max(sbox[3], lbox[3])]
    new_b_2 = [sbox[2], min(sbox[1], lbox[1]), lbox[2], max(sbox[3], lbox[3])]

    if sboxwidth > box_thre: 
        if left_r > box_thre and right_r > box_thre: 
            add_boxes.append(new_b_1)
            add_boxes.append(new_b_2)
            lbox = sbox # keep small one
        elif left_r > box_thre and right_r <= box_thre: 
            add_boxes.append(new_b_1)
            lbox = sbox # keep small one
        elif left_r <= box_thre and right_r > box_thre: 
            add_boxes.append(new_b_2)
            lbox = sbox # keep small one
            # last case is: left_r < 15 and right_r < 15
        else: 
            sbox = lbox # keep large one
    # if the small box is too small, keep the large box
    # the case is sboxwidth < 15
    #else: 
        #sbox = lbox # keep large one
        #edge = min(left_r, right_r, 10)
        #sbox[0] = sbox[0] - edge
        #sbox[2] = sbox[2] + edge 
        
    
    return sbox, lbox, add_boxes


# In[ ]:


# deal with large overlapping boxes

def cleanoverlap(boxes, box_thre = 10):
    
    boxes.sort(key = lambda s: (s[0], s[2]))
    
    add_boxes = []    
    for i in range(len(boxes) - 1):
        for j in range(i+1, len(boxes)):
            
            if boxes[j][0] -2 <= boxes[i][0] and boxes[j][2] + 2 >= boxes[i][2]:
                sbox = boxes[i]
                lbox = boxes[j]
                boxes[i], boxes[j], add_boxes = cleanoverlap_part(sbox, lbox, box_thre)
                
            elif boxes[i][0] -2 <= boxes[j][0] and boxes[i][2] + 2 >= boxes[j][2]:
                sbox = boxes[j]
                lbox = boxes[i]
                boxes[j], boxes[i], add_boxes = cleanoverlap_part(sbox, lbox, box_thre)
                
            else:
                continue

    boxes = boxes + add_boxes
    
    # do cleanning 
    boxes.sort(key = lambda s: (s[0], s[2]))
    boxes_clean = []
    for box in boxes:
        if box not in boxes_clean:
            boxes_clean.append(box)
    
    return boxes_clean


# In[ ]:


### main function to find candidate textboxes:

def cleanbox(boxes, iou_thre_1 = 0.5, box_thre = 15, iou_thre_2 = 0.2):
    
    boxes.sort(key = lambda s: (s[0], s[2]))
    
    # remove same box
    boxes = cleaniou(boxes, iou_thre_1)
    
    # fine cutting
    boxes = cleanoverlap(boxes, box_thre)
    
    # remove overlapping box
    boxes = cleaniou(boxes, iou_thre_2)
    
   
    return boxes


# In[ ]:


# remove empty (or say, sparse) boxes

def rmemptybox(boxes, img, tol = 2):
    
    boxes_clean = []

    boxwidth = []
    
    boxes.sort(key = lambda s: (s[0], s[2]))
    
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        croparea = img[y1 + tol: y2 - tol, x1 + tol: x2 - tol, :]
        ratio1 = np.sum(croparea == 0)/((y2-y1)*(x2-x1)) # 0.10
        ratio2 = np.sum(~croparea == 0)/((y2-y1)*(x2-x1)) # 2.00
        # print(box, '\t', '%.3f, %.3f' % (ratio1, ratio2))
        
        if ratio1 >= 0.1:
            boxes_clean.append(box)
            boxwidth.append(x2-x1)
        if ratio2 <= 1.5 and box not in boxes_clean:
            boxes_clean.append(box)
            boxwidth.append(x2-x1)
            
    #for w in boxwidth:
        #print(w, '\t', np.median(boxwidth), '\t', '%.3f' % (w/np.median(boxwidth)))
        
    return boxes_clean


# In[ ]:


## visualize boxes

def visualizebox(boxes, img, x_tol = 0, y_tol = 0):
    
    t = img.copy()
    for box in boxes:
        cv2.rectangle(t,(box[0] - x_tol, box[1] - y_tol),(box[2] + x_tol, box[3] + y_tol),(0,255,0), 2)

    # Finally show the image
    plt.imshow(t)


# In[ ]:


# find large boxes that may need to be cut 

def findlargebox(boxes, boxwidth, thre = 40):
    
    largebox = []

    for i in range(len(boxes)):
        if boxwidth[i] > thre:
            largebox.append([i, boxes[i], boxwidth[i]])

    return largebox


# In[ ]:


# cut large boxes if necessary

def finecutbox(largebox, img, dilate_iter = 1, erode_iter = 1, 
               col_thre = 0.2, col_sep = 1, start_position = 0, col_min_width = 10):
    
    box_new = []
    
    for box in largebox:
        simg = img[box[1][1]:box[1][3], box[1][0]:box[1][2], :]
        simg_1 = simg.copy()
        
        pre = preprocess(simg_1, dilate_iter, erode_iter)
        h_sum, v_sum = projection(pre, showplot=False)

        cols = findcols(v_sum, col_thre, col_sep, start_position, col_min_width)
        
        if len(cols) > 1:
            for col in cols:
                if col[1] - col[0] >=5:
                    box_new.append([box[0], [box[1][0]+col[0], box[1][1], box[1][0]+col[1], box[1][3]]])
        elif len(cols) == 1:
            col = cols[0]
            if (col[1] - col[0])/box[2] <= 0.4:
                box_new.append([box[0], [box[1][0]+col[0], box[1][1], box[1][0]+col[1], box[1][3]]])
            else:
                box_new.append([box[0], box[1]])
        else:
            box_new.append([box[0], box[1]])
            
    return box_new


# In[ ]:


# replace largebox with smalbox

def usesmallbox(boxes, box_new):
    
    indx = np.unique([k[0] for k in box_new]) # boxes need to be replaced
    final_box = []
    for i in range(len(boxes)):
        if i not in indx:
            final_box.append(boxes[i])
        else:
            for box in box_new:
                if i == box[0]:
                    final_box.append(box[1])
    return final_box


# In[ ]:


# add some margin to finalbox

def addmargin(boxes, img, x_tol = 5, y_tol = 10):
    
    for box in boxes:
        box[0] = max(0, box[0] - x_tol)
        box[1] = max(0, box[1] - y_tol)
        box[2] = min(box[2] + x_tol, img.shape[1])
        box[3] = min(box[3] + y_tol, img.shape[0])
    
    return boxes


# In[ ]:





# In[ ]:


def rough_seg(img, textbox_final, show_seg = False):
    ## rough segment
    imgtext = img[textbox_final[1]:textbox_final[3], textbox_final[0]:textbox_final[2], :]

    pre1 = preprocess(imgtext, dilate_iter = 1, erode_iter = 1)
    pre2 = preprocess(imgtext, dilate_iter = 1, erode_iter = 2)
    pre3 = preprocess(imgtext, dilate_iter = 1, erode_iter = 3)


    boxes1 = findchars(pre1, w_max = 90) # w_max = 90,  h_max = pre1.shape[0]
    boxes2 = findchars(pre2, w_max = 90)
    boxes3 = findchars(pre3, w_max = 90)

    boxes = boxes1 + boxes2 + boxes3
    boxes = cleanbox(boxes, iou_thre_1=0.5, box_thre = 15, iou_thre_2=0.15)

    boxes = rmemptybox(boxes, imgtext, tol = 2)
    
    boxwidth = []
    for i, box in enumerate(boxes):
        boxwidth.append(box[2] - box[0])

    if show_seg:
        visualizebox(boxes, imgtext, x_tol = 5, y_tol = 5) # if want show
        plt.show()
        
    return imgtext, boxes, boxwidth


# In[ ]:


def fine_seg(imgtext, boxes, boxwidth, show_seg = False):
    largebox = findlargebox(boxes, boxwidth)
    box_new = finecutbox(largebox, imgtext)
    final_box = usesmallbox(boxes, box_new)

    final_box_margin = copy.deepcopy(final_box)
    final_box_margin = addmargin(final_box_margin, imgtext, x_tol = 3, y_tol = 5)
    
    if show_seg:
        visualizebox(final_box_margin, imgtext)
    
    return final_box_margin


# In[ ]:





# In[1]:


## following sections for cnn model


# In[ ]:


def getcharimg(img, boxes, prepare = False, dilate_iter = None, erode_iter = None, threshval = 75, showplot=False):
    
    charimg = []
    original_shape = []
    for box in boxes:
        if prepare:
            new_img = preprocess(img[box[1]:box[3], box[0]:box[2], :], dilate_iter, erode_iter)
            charimg.append(new_img)
            original_shape.append(new_img.shape)
        else: 
            new_img = cv2.cvtColor(img[box[1]:box[3], box[0]:box[2], :],cv2.COLOR_BGR2GRAY)
            new_img = cv2.threshold(new_img, threshval, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
            charimg.append(new_img)
            original_shape.append(new_img.shape)
    
    if showplot:
        
        fig, axes = plt.subplots(3, 4, figsize = (5, 5))
        
        for i in range(len(charimg)):
            r_id, c_id = i//4, i%4
            axes[r_id, c_id].imshow(charimg[i], cmap = 'gray')
            
    return charimg, original_shape


# In[ ]:


def cleancharimg(predict_img, showplot = False):
    predict_img_clean = []
    for img in predict_img: 
        img = cv2.dilate(~img,  None, 1)
        img = cv2.erode(img, None, 1)
        # append
        predict_img_clean.append(~img)

    if showplot:
        fig, axes = plt.subplots(3, 4, figsize = (5, 5))
        for i in range(len(predict_img_clean)):
            r_id, c_id = i//4, i%4
            axes[r_id, c_id].imshow(predict_img_clean[i], cmap = 'gray')    
            
    return predict_img_clean


# In[ ]:





# In[ ]:


# resize and normalize

def resize_norm(predict_img, target_h = 28, target_w = 28, showplot = False):
    X_predict = []
    for img in predict_img:
        cur_h, cur_w = img.shape[0], img.shape[1]
        if cur_h > cur_w:
            ratio = cur_h / target_h
            new_img = imresize(img, [target_h, int(cur_w/ratio)])
            pad_left = int((target_w - cur_w/ratio)/2)
            pad_right = target_w - new_img.shape[1] - pad_left
            # add to left and right
            new_img = np.pad(new_img, ((0, 0), (pad_left, pad_right)), 'constant')
        else:
            ratio = cur_w / target_w
            new_img = imresize(img, [int(cur_h/ratio), target_w])
            pad_top = int((target_h - cur_h/ratio)/2)
            pad_bottom = target_h - new_img.shape[0] - pad_top
            # add to top and height
            new_img = np.pad(new_img, ((pad_top, pad_bottom), (0, 0)), 'constant')
        
        # new_img = imresize(img, [target_h, target_w])
        new_img = new_img/255
        
        X_predict.append(new_img[:, :, np.newaxis])
        
    X_predict = np.stack(X_predict, axis=0)
    
    if showplot:
        fig, axes = plt.subplots(3, 4, figsize = (5, 5))

        for i in range(len(X_predict)):
            r_id, c_id = i//4, i%4
            axes[r_id, c_id].imshow(X_predict[i][:, :, 0], cmap = 'gray')
            
    return X_predict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




