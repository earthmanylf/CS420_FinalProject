import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

# batch normalization layer with an optional activation layer
def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act:
        x = Activation('relu')(x)
    return x

# conv layer with a batch normalization layer
def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(conv)
    return conv

def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, kernel_size, padding, strides)
    res = conv_block(res, filters, kernel_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUnet(pretrained_weights=None, input_size=(256,256,1)):
    inputs = Input(input_size)
    # Down
    d1 = stem(inputs, 16)
    d2 = residual_block(d1, 32, strides=2)
    d3 = residual_block(d2, 64, strides=2)
    d4 = residual_block(d3, 128, strides=2)
    d5 = residual_block(d4, 256, strides=2)

    b0 = conv_block(d5, 256)
    b1 = conv_block(b0, 256)

    # Up
    u1 = upsample_concat_block(b1, d4)
    r1 = residual_block(u1, 256)
    u2 = upsample_concat_block(r1, d3)
    r2 = residual_block(u2, 128)
    u3 = upsample_concat_block(r2, d2)
    r3 = residual_block(u3, 64)
    u4 = upsample_concat_block(r3, d1)
    r4 = residual_block(u4, 32)
    outputs = Conv2D(1, 1, activation='sigmoid')(r4)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


