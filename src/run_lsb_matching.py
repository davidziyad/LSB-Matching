import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from .net import FrequencyNet
from .lsb_matching import encode, decode
from .lsb import LSBMatcher


""" READ COVER IMAGE """
rows = 128
cols = 128
image = cv.imread('im.jpg')
image_ = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image_, (rows, cols))
image = np.array(image)


""" DEFINE MESSAGE TO EMBED """
message_to_encode = 'a' * 50000

assert len(message_to_encode) * 8 <= image_.shape[0] * image_.shape[1]


""" COMPUTE FREQUENCY IMAGE """
num_latent_images = 16

frequency_net = FrequencyNet(
    image,
    num_latent_images
)

frequency_net.train()

frequency_image = frequency_net.compute_frequency_image()
frequency_image = cv.resize(frequency_image, (image_.shape[1], image_.shape[0]))
cv.imwrite('results/freq_image.jpg', frequency_image)

""" EMBED MESSAGE """
lsb_matcher = LSBMatcher()
stego_image = lsb_matcher.embed(message_to_encode, image_, frequency_image)


""" RECOVER MESSAGE """
# decoded_message = decode(stego_image, original_pixel_values, indices)


""" VALIDATE RESULTS """
# print('Original Message:', message_to_encode)
# print('Recovered Message:', decoded_message)

# assert message_to_encode == decoded_message


""" PLOT RESULTS """
encoded_image = stego_image.reshape((image_.shape[0], image_.shape[1], 3))
fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax_1.imshow(image_)
ax_2.imshow(encoded_image)
plt.show()

""" PERSIST RESULTS """

rgb_image = cv.cvtColor(image_, cv.COLOR_BGR2RGB)
rgb_stego = cv.cvtColor(stego_image.reshape((image_.shape[0], image_.shape[1], 3)), cv.COLOR_BGR2RGB)
cv.imwrite('results/cover_image.jpg', rgb_image)
cv.imwrite('results/embed_image.jpg', rgb_stego)
