import numpy as np


class LSBMatcher:

    def __init__(self):
        pass

    def _message2binary(self, msg):
        return ''.join(format(ord(x), 'b') for x in msg)

    def _dec2bin(self, decimal_number):
        return '{0:b}'.format(decimal_number)

    def _bin2dec(self, binary_number):
        return int(binary_number, 2)

    def _lsb(self, binary_number):
        return binary_number[-1]

    def _f(self, y1, y2):
        return self._lsb(
            self._dec2bin(
                int(
                    np.floor(
                        self._bin2dec(y1) / 2.0 + self._bin2dec(y2)
                    )
                )
            )
        )

    def _embed_bit(self, x1, x2, m1, m2):
        """ All arguments are in binary form.
        """

        if m1 == self._lsb(x1):
            if m2 != self._f(x1, x2):
                y2 = self._bin2dec(x2) + 1
            else:
                y2 = self._bin2dec(x2)

            y1 = self._bin2dec(x1)
        else:
            if m2 == self._f(self._dec2bin(self._bin2dec(x1) - 1), x2):
                y1 = self._bin2dec(x1) - 1
            else:
                y1 = self._bin2dec(x1) + 1
            y2 = self._bin2dec(x2)

        return y1, y2

    def embed(self, msg, cover_image, frequency_image):
        cover_image_vec = cover_image.ravel()
        frequency_image_vec = frequency_image.ravel()

        assert len(cover_image_vec) == len(frequency_image_vec)

        binary_message = self._message2binary(msg)

        assert len(cover_image_vec) >= len(binary_message)

        ranked_pixels = frequency_image_vec.argsort()

        stego_image = cover_image_vec.copy()

        for i in range(len(ranked_pixels) - 1):
            if i + 1 == len(binary_message):
                break

            x1 = cover_image_vec[ranked_pixels[i]]
            x2 = cover_image_vec[ranked_pixels[i + 1]]

            if x1 in [0, 255] or x2 in [0, 255]:
                continue

            m1 = binary_message[i]
            m2 = binary_message[i + 1]

            y1, y2 = self._embed_bit(self._dec2bin(x1), self._dec2bin(x2), m1, m2)

            stego_image[ranked_pixels[i]] = y1
            stego_image[ranked_pixels[i + 1]] = y2

        return stego_image

    def extract(self):
        pass
