from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import cv2

N = 10000
characters = string.digits
width, height, n_len, n_class = 210, 80, 6, len(characters)
obj = ImageCaptcha(width=width, height=height)

for i in range(N):

    random_str = ''.join([random.choice(characters) for j in range(n_len)])
    img = obj.generate_image(random_str)
    img1 = np.array(img)
    cv2.imwrite("img/" + str(i).zfill(4) + '_' + random_str + '.jpg', img1)

