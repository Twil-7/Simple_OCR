from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import string


# 总字符类别数： 10种数字
def get_model():

    char_class = string.digits
    width, height, n_len, n_class = 210, 80, 6, len(char_class)
    input_tensor = Input((height, width, 3))

    x = input_tensor
    for i in range(4):
        for j in range(2):

            # 实现两个valid类型的卷积运算
            x = Conv2D(32 * 2 ** i, (3, 1), activation='relu')(x)
            x = BatchNormalization()(x)

            x = Conv2D(32 * 2 ** i, (1, 3), activation='relu')(x)
            x = BatchNormalization()(x)

        # 下采样
        x = Conv2D(32 * 2 ** i, 2, 2, activation='relu')(x)
        x = BatchNormalization()(x)

    # 此时输出结构： (None, 1, 9, 256)

    out = []
    # 6个输出尺度，相互独立
    for i in range(n_len):

        out_branch = Conv2D(n_class, (1, 9))(x)
        out_branch = Reshape((n_class,))(out_branch)
        out_branch = Activation('softmax', name='c%d' % (i + 1))(out_branch)
        out.append(out_branch)

    model = Model(inputs=input_tensor, outputs=out)
    model.summary()

    return model

