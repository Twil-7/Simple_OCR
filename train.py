import cv2
import os
import random
import numpy as np
import string
from tensorflow.keras.utils import *
import math
from ocr_model import get_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


char_class = string.digits
width, height, n_len, n_class = 210, 80, 6, len(char_class)
char_list = list(char_class)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class SequenceData(Sequence):

    def __init__(self, data_x, data_y, batch_size):
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.indexes = np.arange(len(self.data_x))

    def __len__(self):
        return math.floor(len(self.data_x) / float(self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, idx):

        batch_index = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.data_x[k] for k in batch_index]
        batch_y = [self.data_y[k] for k in batch_index]

        x = np.zeros((self.batch_size, height, width, 3))
        y = [np.zeros((self.batch_size, n_class)) for k in range(n_len)]    # n_len 个 (batch_size, n_class)

        for i in range(self.batch_size):

            img = cv2.imread(batch_x[i])
            img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            img2 = img1 / 255
            x[i, :, :, :] = img2

            for j in range(n_len):

                char = batch_y[i][j]
                char_index = char_class.find(char)
                y[j][i, char_index] = 1

        return x, y


# create model and train and save
def train_network(train_generator, validation_generator, epoch):

    model = get_model()

    adam = Adam(lr=1e-3, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}-train_loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights('first_weights.hdf5')


def load_network_then_train(train_generator, validation_generator, epoch, input_name, output_name):

    model = get_model()
    model.load_weights(input_name)
    print('网络层总数为：', len(model.layers))  # 175

    adam = Adam(lr=1e-4, amsgrad=True)
    log_dir = "Logs/"
    checkpoint = ModelCheckpoint(log_dir + 'epoch{epoch:03d}-train_loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint]
    )

    model.save_weights(output_name)



