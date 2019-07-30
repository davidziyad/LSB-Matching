# LSB Matching
import math
import numpy as np

def string_to_binary(st):
    return ''.join(format(ord(x), 'b') for x in st)

def decimal_to_binary(x):
    return int(bin(x)[2:])
    
def binary_to_decimal(x):
    return int(str(x), 2)
    
def LSB(x):
    # x is a integer value not binary value
    return x & 1
    
def function_f(y1_dec, y2_dec):
    #y1 and y2 are decimal values    
    output = LSB(int(math.floor(y1_dec/2) + y2_dec))
    return output

def LSB_Matching_encode(image_array, pixel_ind, message): #flatten image to be 1D array to get pixel values
    
    if len(pixel_ind) <= len(message):
            raise Exception("Not enough pixels to encode message")
    
    original_pixel_values = image_array[pixel_ind[:len(message)]]
    new_image = image_array
    
    for i in range(0, len(message)-1, 2):
        y1 = 0
        y2 = 0
        
        if int(message[i]) == LSB(image_array[pixel_ind[i]]):
            if int(message[i+1]) != function_f(image_array[pixel_ind[i]], image_array[pixel_ind[i+1]]):
                y2 = image_array[pixel_ind[i+1]] + 1 
            else:
                y2 = image_array[pixel_ind[i+1]]
            
            y1 = image_array[pixel_ind[i]]
        else:
#            pdb.set_trace()
            if int(message[i+1]) == function_f(image_array[pixel_ind[i]]-1, image_array[pixel_ind[i+1]]): 
                y1 = image_array[pixel_ind[i]] - 1
            else:
                y1 = image_array[pixel_ind[i]] + 1 
            
            y2 = image_array[pixel_ind[i+1]]
        
        new_image[pixel_ind[i]] = y1
        new_image[pixel_ind[i+1]] = y2
        
    return image_array, original_pixel_values, pixel_ind[:len(message)]

def LSB_Matching_decode(encoded_image, original_values, indices):
    message = ''
#    pdb.set_trace()
    for i in range(0, len(original_values)-1, 2):
        first_value_enc = encoded_image[indices[i]]
        second_value_enc = encoded_image[indices[i+1]]
        
        first_orig = original_values[i]
        second_orig = original_values[i+1]
        
        message_i = LSB(first_orig)
        
        if first_orig == first_value_enc:    
            message = message + str(message_i)
            
            message_i_plus_1 = function_f(first_orig, second_orig)
            if (second_value_enc-1) == second_orig:    
                piece_message = 1 - message_i_plus_1
                message = message + str(piece_message)
            else:
                message = message + str(message_i_plus_1)
                
        else:
            message = message + str(1-message_i)
            message_i_plus_1 = function_f(first_orig-1, second_orig)
            if first_orig == (first_value_enc + 1):
                message = message + str(message_i_plus_1)
            else:
                message = message + str(1 - message_i_plus_1)
            
    return message
