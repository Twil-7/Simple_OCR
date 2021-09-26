import cv2
from ocr_model import get_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import string
from get_data import make_data


char_class = string.digits
width, height, n_len, n_class = 210, 80, 6, len(char_class)
char_list = list(char_class)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def predict_sequence(test_x, test_y):

    predict_model = get_model()
    predict_model.load_weights('best_val_loss1.982.h5')

    acc_count = 0     # 统计正确的序列个数
    char_count = 0    # 统计正确的字符个数

    for i in range(len(test_x)):

        img = cv2.imread(test_x[i])
        img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img2 = img1 / 255
        img3 = img2[np.newaxis, :, :, :]

        y_pre = predict_model.predict(img3)
        y_pre1 = np.array(y_pre)
        y_pre2 = ''

        for j in range(n_len):

            vector = y_pre1[j, 0, :]
            index = int(np.argmax(vector))
            char = char_list[index]

            if char == test_y[i][j]:
                char_count = char_count + 1

            y_pre2 = y_pre2 + char

        if y_pre2 == test_y[i]:
            acc_count = acc_count + 1

        # print(y_pre2)
        # cv2.namedWindow("img2")
        # cv2.imshow("img2", img2)
        # cv2.waitKey(0)

    print('sequence recognition accuracy : ', acc_count / len(test_x))
    print('char recognition accuracy : ', char_count / (len(test_x) * n_len))


