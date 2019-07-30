# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:40:38 2019

@author: ziyad
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ae import get_image_to_encode
from LSB_Matching import LSB_Matching_encode, string_to_binary, LSB_Matching_decode


rows = 128
cols = 128
image = cv2.imread('images.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (rows,cols))
image = np.array(image)

string_to_encode = 'This is a message I would like to encode if possible!'

# get high freqency image
resultant_image = get_image_to_encode(image)

# flatten to 1D array
flat_image = resultant_image.flatten()
image_to_encode = image.flatten()

# get all indicies 
indices = np.argsort(flat_image)

# binarise message
string_binary = string_to_binary(string_to_encode)
encoded_image, original_pixel_values, indices = LSB_Matching_encode(image_to_encode, indices, string_binary)

decoded_message = LSB_Matching_decode(encoded_image, original_pixel_values, indices)

print string_binary == decoded_message


# Decode binary message
# need to fix
#def bin2text(s): 
#    return "".join([chr(int(s[i:i+8],2)) for i in xrange(0,len(s),8)])




# plot results
encoded_image = encoded_image.reshape((rows,cols,3))
fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax_1.imshow(image)
ax_2.imshow(encoded_image)
