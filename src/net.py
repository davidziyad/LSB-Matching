
import numpy as np
import cv2 as cv

from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add


class FrequencyNet:

    def __init__(self, cover_image, num_latent_images):
        self.cover_image = cover_image
        self.num_latent_images = num_latent_images

        self.frequency_net = None
        self.encoder_net = None
        self.frequency_image = None

        self._create_model()

    def _create_model(self):
        image_i = Input(shape=(128, 128, 3))
        noise_i = Input(shape=(128, 128, 3))

        stream = Add()([image_i, noise_i])
        stream = Conv2D(16, (3, 3), activation='relu', padding='same')(stream)
        stream = MaxPooling2D()(stream)
        stream = Conv2D(32, (3, 3), activation='relu', padding='same')(stream)
        stream = MaxPooling2D()(stream)
        stream = Conv2D(64, (3, 3), activation='relu', padding='same')(stream)
        stream = MaxPooling2D()(stream)

        latent = Conv2D(self.num_latent_images, (3, 3), activation='relu', padding='same', name='latent')(stream)

        o_stream = UpSampling2D()(latent)
        o_stream = Conv2D(64, (3, 3), activation='relu', padding='same')(o_stream)
        o_stream = UpSampling2D()(o_stream)
        o_stream = Conv2D(32, (3, 3), activation='relu', padding='same')(o_stream)
        o_stream = UpSampling2D()(o_stream)
        o_stream = Conv2D(16, (3, 3), activation='relu', padding='same')(o_stream)
        output = Conv2D(3, (3, 3), activation='relu', padding='same')(o_stream)

        self.frequency_net = Model([image_i, noise_i], output)
        self.frequency_net.compile(loss='mean_squared_error', optimizer='adam')

        self.encoder_net = Model([image_i, noise_i], latent)

    def _generator(self, im, batch_size=4):
        while True:
            X_image = np.array([im for _ in range(batch_size)]) / 255.0
            X_noise = np.array([np.random.randint(0, 5, (128, 128, 3)) for _ in range(batch_size)]) / 255.0

            yield [X_image, X_noise], X_image

    def train(self):
        self.frequency_net.fit_generator(self._generator(self.cover_image), epochs=128, steps_per_epoch=1)

    def _compute_latent_images(self):
        X_te_image = np.expand_dims(self.cover_image, axis=0) / 255.0
        X_te_noise = np.expand_dims(np.zeros((128, 128, 3)).astype('float'), axis=0)
        return self.encoder_net.predict([X_te_image, X_te_noise])

    def compute_frequency_image(self):
        a = []

        pred = self._compute_latent_images()

        for j in range(self.num_latent_images):
            e = pred[0, :, :, j]
            ff = cv.resize(e, (128, 128))
            ff = cv.cvtColor(ff, cv.COLOR_GRAY2RGB)
            a.append(ff)
        a = np.array(a)
        a = a.mean(0)

        self.frequency_image = (a / a.max() * 255.0).astype('uint8')
