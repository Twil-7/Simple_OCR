import cv2
from ocr_model import get_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import string


char_class = string.digits
width, height, n_len, n_class = 210, 80, 6, len(char_class)
char_list = list(char_class)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 获取特定名称的网络层
def get_output_layer(model1, layer_name):

    layer_dict = dict([(layer.name, layer) for layer in model1.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_region(new_model, img, conv_name, k):

    aug_img = img[np.newaxis, :, :, :]
    y_pre = new_model.predict(aug_img)

    add_result = y_pre[-1]    # (1, 1, 9, 256)
    original_result = y_pre[:-1]    # 6 个 (1, 36)

    # 取出输出层某个分类器分支，并将其权重提取出来
    weights_layer = get_output_layer(new_model, conv_name)
    class_weights = weights_layer.get_weights()[0]    # (1, 9, 256, 36)

    # 从这个分类器权重中，将其预测标签的权重进一步提取出来
    predict_index = np.argmax(np.array(original_result), axis=2)[:, 0]    # (6,)
    class_weights_w = class_weights[:, :, :, predict_index[k]]    # (1, 9, 256)

    # 将分类器的权重与卷积模型的特征相乘后再求和
    cam = np.sum(add_result[0] * class_weights_w, axis=-1)    # (1, 9)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0)] = 0

    cv2.imwrite('heatmap' + str(k) + '.jpg', heatmap)


def plot_heatmap():

    model = get_model()
    final_layer = get_output_layer(model, 'batch_normalization_19')  # (None, 1, 9, 256)

    out = model.output
    out.append(final_layer.output)
    new_model = Model(inputs=model.input, outputs=out, name='new_model')
    new_model.load_weights('best_val_loss1.982.h5')

    img = cv2.imread('039796.jpg')
    img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img2 = img1 / 255

    # 对第1-6个分类器分支，分别进行可视化
    visualize_region(new_model, img2, 'conv2d_20', 0)
    visualize_region(new_model, img2, 'conv2d_21', 1)
    visualize_region(new_model, img2, 'conv2d_22', 2)
    visualize_region(new_model, img2, 'conv2d_23', 3)
    visualize_region(new_model, img2, 'conv2d_24', 4)
    visualize_region(new_model, img2, 'conv2d_25', 5)


plot_heatmap()

