import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from unet import Unet
from resunet import ResUnet
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from acc import compute_all_acc_no_file

train_data_file = 'dataset/new_train_set'
test_data_file = 'dataset/new_test_set'

# parameters of data augmentation
data_gen_list = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

def AdjustData(data, label):
    data = data / 255
    label = label / 255
    label[label>0.5] = 1
    label[label<=0.5] = 0
    return data, label

def GetTrainData(train_data_path):
    data_imggen = ImageDataGenerator(**data_gen_list)
    label_imggen = ImageDataGenerator(**data_gen_list)
    data_generator = data_imggen.flow_from_directory(train_data_path,
                                                     classes=['train_img'],
                                                     class_mode=None,
                                                     color_mode='grayscale',
                                                     target_size=(256, 256),
                                                     batch_size=5,
                                                     save_to_dir=None,
                                                     seed=1)
    label_generator = label_imggen.flow_from_directory(train_data_path,
                                                     classes=['train_label'],
                                                     class_mode=None,
                                                     color_mode='grayscale',
                                                     target_size=(256, 256),
                                                     batch_size=5,
                                                     save_to_dir=None,
                                                     seed=1)
    train_generator = zip(data_generator, label_generator)
    for (data, label) in train_generator:
        data, label = AdjustData(data, label)
        yield data, label

def GetTestData(test_data_path, num_image=5):
    imgs = []
    labels = []
    for i in range(num_image):
        img = io.imread(test_data_path + '/test_img/' + str(i) + '.png', as_gray=True)
        img = img / 255
        img = trans.resize(img, (256, 256))
        img = np.reshape(img, img.shape + (1,))
        imgs.append(img)
    imgs = np.array(imgs)
    for i in range(num_image):
        label = io.imread(test_data_path + '/test_label/' + str(i) + '.png', as_gray=True)
        label = label / 255
        label = trans.resize(label, (256, 256))
        label[label>0.5] = 1
        label[label<=0.5] = 0
        labels.append(label)
    labels = np.array(labels)
    return imgs, labels

# mode--'unet' or 'resunet'
def train(mode):
    trainGen = GetTrainData(train_data_file)
    if mode == 'unet':
        model = ResUnet(pretrained_weights='unet.h5')
        checkpoint = ModelCheckpoint('unet.h5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(trainGen, 300, 10, callbacks=[checkpoint])
    elif mode == 'resunet':
        model = ResUnet(pretrained_weights='resunet.h5')
        checkpoint = ModelCheckpoint('resunet.h5', monitor='loss', verbose=1, save_best_only=True)
        model.fit_generator(trainGen, 300, 10, callbacks=[checkpoint])

# modelfile--'unet.h5' or 'resunet.h5'
def test(modelfile):
    model = load_model(modelfile)
    test, label = GetTestData(test_data_file)
    test_eval = model.predict(test)
    for k in range(5):
        for i in range(256):
            for j in range(256):
                if test_eval[k][i][j][0] <= 0.5:
                    test_eval[k][i][j][0] = 0
                elif test_eval[k][i][j][0] > 0.5:
                    test_eval[k][i][j][0] = 1
    test_eval = test_eval.reshape((5,256,256))

    compute_all_acc_no_file(test_eval, label)

if __name__ == '__main__':
    train('unet')



