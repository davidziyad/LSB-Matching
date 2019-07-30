import cv2 as cv
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add, Input
from keras.models import Model, load_model
from keras import backend as K
import matplotlib.pyplot as plt

def imshow(i, cmap='gray'):
    plt.figure()
    plt.imshow(i, cmap=cmap)
    plt.show()

def gen(im, batch_size=4):
    while True:
        
        X_image = np.array([im for _ in range(batch_size)]) / 255.0
        X_noise = np.array([np.random.randint(0, 5, (128, 128, 3)) for _ in range(batch_size)]) / 255.0
        
        yield [X_image, X_noise], X_image

def create_AE(image_i, noise_i, L):    
    
    stream = Add()([image_i, noise_i])
    stream = Conv2D(16, (3, 3), activation='relu', padding='same')(stream)
    stream = MaxPooling2D()(stream)
    stream = Conv2D(32, (3, 3), activation='relu', padding='same')(stream)
    stream = MaxPooling2D()(stream)
    stream = Conv2D(64, (3, 3), activation='relu', padding='same')(stream)
    stream = MaxPooling2D()(stream)

    latent = Conv2D(L, (3, 3), activation='relu', padding='same', name='latent')(stream)
    
    o_stream = UpSampling2D()(latent)
    o_stream = Conv2D(64, (3, 3), activation='relu', padding='same')(o_stream)
    o_stream = UpSampling2D()(o_stream)
    o_stream = Conv2D(32, (3, 3), activation='relu', padding='same')(o_stream)
    o_stream = UpSampling2D()(o_stream)
    o_stream = Conv2D(16, (3, 3), activation='relu', padding='same')(o_stream)
    output = Conv2D(3, (3, 3), activation='relu', padding='same')(o_stream)
    
    m = Model([image_i, noise_i], output)
    m.compile(loss='mean_squared_error', optimizer='adam')

    return m, latent

def predict_latent(im, L, steps_per_epochs=1):
    
    image_i = Input(shape=(128, 128, 3))
    noise_i = Input(shape=(128, 128, 3))
    
    m, latent = create_AE(image_i, noise_i, L)

    m.fit_generator(gen(im), epochs=128, steps_per_epoch=steps_per_epochs)
    
    mm = Model([image_i, noise_i], latent)
    X_te_image = np.expand_dims(im, axis=0) / 255.0
    X_te_noise = np.expand_dims(np.zeros((128, 128, 3)).astype('float'), axis=0)
    pred = mm.predict([X_te_image, X_te_noise])
    
    return pred

def get_mean_latent_image(im, L):     
    a = []

    pred = predict_latent(im, L)    
    
    for j in range(L):
        e = pred[0, :, :, j]
        ff = cv.resize(e, (128, 128))
        ff = cv.cvtColor(ff, cv.COLOR_GRAY2RGB)
        a.append(ff)
    a = np.array(a)
    a = a.mean(0)
    
    return a
    
def convert_to_image_range(im, L):
    
#    vis = cv.multiply(im.astype('float32'), a)
#    vis = vis / vis.max()
    mean_image = get_mean_latent_image(im, L)
    image = (mean_image / mean_image.max() * 255.0).astype('uint8')
    
    return image

def get_image_to_encode(im):    
    L = 16
    resultant_image = convert_to_image_range(im, L)
    
    return resultant_image


im = cv.imread('images.jpeg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
im = cv.resize(im, (128, 128))
image = get_image_to_encode(im)
#added_image = cv.addWeighted(im, 0.1, resultant_image, 0.9, 0)
#imshow(added_image)
#imshow(resultant_image)
#imshow(im)
