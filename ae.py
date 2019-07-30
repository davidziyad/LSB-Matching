import cv2 as cv
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add, Input
from keras.models import Model, load_model
from keras import backend as K
import matplotlib.pyplot as plt

im = cv.imread('images.jpeg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
im = cv.resize(im, (128, 128))

def imshow(i, cmap='gray'):
    plt.figure()
    plt.imshow(i, cmap=cmap)
    plt.show()

def gen():
    while True:
        
        X_image = np.array([im for _ in range(batch_size)]) / 255.0
        X_noise = np.array([np.random.randint(0, 5, (128, 128, 3)) for _ in range(batch_size)]) / 255.0
        
        yield [X_image, X_noise], X_image


image_i = Input(shape=(128, 128, 3))

noise_i = Input(shape=(128, 128, 3))


stream = Add()([image_i, noise_i])
stream = Conv2D(16, (3, 3), activation='relu', padding='same')(stream)
stream = MaxPooling2D()(stream)
stream = Conv2D(32, (3, 3), activation='relu', padding='same')(stream)
stream = MaxPooling2D()(stream)
stream = Conv2D(64, (3, 3), activation='relu', padding='same')(stream)
stream = MaxPooling2D()(stream)

L = 16
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

batch_size = 4
steps_per_epochs = 1
m.fit_generator(gen(), epochs=128, steps_per_epoch=steps_per_epochs)

mm = Model([image_i, noise_i], latent)
X_te_image = np.expand_dims(im, axis=0) / 255.0
X_te_noise = np.expand_dims(np.zeros((128, 128, 3)).astype('float'), axis=0)
pred = mm.predict([X_te_image, X_te_noise])

a = []
for j in range(L):
    e = pred[0, :, :, j]
    ff = cv.resize(e, (128, 128))
    ff = cv.cvtColor(ff, cv.COLOR_GRAY2RGB)
    a.append(ff)
a = np.array(a)
a = a.mean(0)
    
vis = cv.multiply(im.astype('float32'), a)
vis = vis / vis.max()


a = (a / a.max() * 255.0).astype('uint8')

added_image = cv.addWeighted(im, 0.1, a, 0.9, 0)
imshow(added_image)

imshow(a)

imshow(im)
