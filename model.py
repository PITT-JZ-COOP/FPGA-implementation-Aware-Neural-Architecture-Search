'''
Author: Weiwen Jiang, Xinyi Zhang
'''
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D

# generic model design
def model_fn(actions):
    # unpack the actions from the list
    kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions

    ip = Input(shape=( 28, 28, 1))
    x = Conv2D(filters_1, (kernel_1, kernel_1), strides=(1, 1), padding='same', activation='relu')(ip)
    x = Conv2D(filters_2, (kernel_2, kernel_2), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_3, (kernel_3, kernel_3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_4, (kernel_4, kernel_4), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_5, (kernel_5, kernel_5), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_6, (kernel_6, kernel_6), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_7, (kernel_7, kernel_7), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_8, (kernel_8, kernel_8), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_9, (kernel_9, kernel_9), strides=(1, 1), padding='same', activation='relu')(x)
    #x = Conv2D(filters_10, (kernel_10, kernel_10), strides=(1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model
