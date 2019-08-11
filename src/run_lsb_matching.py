import numpy as np
import cv2
import matplotlib.pyplot as plt

from .net import FrequencyNet
from .lsb_matching import encode, decode


""" READ COVER IMAGE """
rows = 128
cols = 128
image = cv2.imread('images.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (rows,cols))
image = np.array(image)


""" DEFINE MESSAGE TO EMBED """
message_to_encode = 'This is a message I would like to encode if possible!'


""" COMPUTE FREQUENCY IMAGE """
num_latent_images = 16

frequency_net = FrequencyNet(
    image,
    num_latent_images
)

frequency_net.train()

frequency_image = frequency_net.compute_frequency_image()


""" EMBED MESSAGE """
flat_image = frequency_image.flatten()
image_to_encode = image.flatten()
indices = np.argsort(flat_image)

stego_image, original_pixel_values, indices = \
    encode(image_to_encode, indices, message_to_encode)


""" RECOVER MESSAGE """
decoded_message = decode(stego_image, original_pixel_values, indices)


""" VALIDATE RESULTS """
print('Original Message:', message_to_encode)
print('Recovered Message:', decoded_message)

assert message_to_encode == decoded_message


""" PLOT RESULTS """
encoded_image = stego_image.reshape((rows, cols, 3))
fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax_1.imshow(image)
ax_2.imshow(encoded_image)