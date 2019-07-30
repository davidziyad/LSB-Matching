# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:43:57 2019

@author: ziyad
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def MSE_metric(original_image, encoded_image):
    # We want a low MSE value which corresponds to a low error
    MSE = np.mean((original_image - encoded_image) ** 2)
    
    return MSE
    
def PSNR_metric(original_image, encoded_image):
    MSE = MSE_metric(original_image, encoded_image)
    if MSE == 0:
        return 100
    # R = 1 if images are double precision floating poits
    # R = 255 is it is 8-bit unsigned integer data type
    # Assume R is 8bit unsigned for now
    R = 255.0 
    PSNR = 10 * math.log10( (R**2)  / MSE )
    
    return PSNR
    
import cv2 as cv

im = cv.imread('images.jpeg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
im = cv.resize(im, (128, 128))
noise = np.array(np.random.randint(-1, 1, (128, 128, 3)))

im_noise = im + noise

MSE = MSE_metric(im, im_noise) 
PSNR = PSNR_metric(im, im_noise) 

print "MSE: ", MSE
print "PSNR: ", PSNR

fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True)
ax_1.imshow(im)
ax_2.imshow(im_noise)