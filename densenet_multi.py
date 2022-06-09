import tensorflow.keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy, mse
import numpy as np

l2_lambda = 0.0002
DropP = 0.3


def dense_block(conv):
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(conv)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    y = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(x)
    y = tensorflow.keras.layers.BatchNormalization()(y)
    merge = tensorflow.keras.layers.concatenate([x, y])
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    merge = tensorflow.keras.layers.concatenate([merge, x])
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    return x


def denser_block(conv, layers, filters):
    x = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(conv)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    y = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(x)
    y = tensorflow.keras.layers.BatchNormalization()(y)
    merge = tensorflow.keras.layers.concatenate([x, y])
    for z in range(layers - 2):
        x = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu',
                                           padding='same',
                                           kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        merge = tensorflow.keras.layers.concatenate([merge, x])
    return x


def multi_dense_model():
    input_big = tensorflow.keras.layers.Input(shape=(1, 64, 64, 64))
    prepool = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(input_big)

    pool1 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(dense_block(prepool))
    pool1 = tensorflow.keras.layers.Dropout(DropP)(pool1)
    flatten1 = tensorflow.keras.layers.Flatten()(pool1)
    output1 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten1)

    pool2 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool1, 10, 12))
    pool2 = tensorflow.keras.layers.Dropout(DropP)(pool2)
    flatten2 = tensorflow.keras.layers.Flatten()(pool2)
    output2 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten2)

    pool3 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool2, 20, 12))
    pool3 = tensorflow.keras.layers.Dropout(DropP)(pool3)
    flatten3 = tensorflow.keras.layers.Flatten()(pool3)
    output3 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten3)

    pool4 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool3, 20, 24))
    pool4 = tensorflow.keras.layers.Dropout(DropP)(pool4)
    flatten4 = tensorflow.keras.layers.Flatten()(pool4)
    output4 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten4)

    pool5 = denser_block(pool4, 20, 48)

    input_small = tensorflow.keras.layers.Input(shape=(1, 32, 32, 32))

    pool6 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(dense_block(input_small))
    pool6 = tensorflow.keras.layers.Dropout(DropP)(pool6)
    flatten6 = tensorflow.keras.layers.Flatten()(pool6)
    output6 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten6)

    pool7 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool6, 10, 12))
    pool7 = tensorflow.keras.layers.Dropout(DropP)(pool7)
    flatten7 = tensorflow.keras.layers.Flatten()(pool7)
    output7 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten7)

    pool8 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool7, 20, 12))
    pool8 = tensorflow.keras.layers.Dropout(DropP)(pool8)
    flatten8 = tensorflow.keras.layers.Flatten()(pool8)
    output8 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten8)

    pool9 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool8, 20, 24))
    pool9 = tensorflow.keras.layers.Dropout(DropP)(pool9)
    flatten9 = tensorflow.keras.layers.Flatten()(pool9)
    output9 = tensorflow.keras.layers.Dense(1, activation='sigmoid')(flatten9)

    pool10 = denser_block(pool9, 20, 48)

    merge_both_blocks = tensorflow.keras.layers.concatenate([pool10, pool5])
    flatten10 = tensorflow.keras.layers.Flatten()(merge_both_blocks)

    final_merge = tensorflow.keras.layers.concatenate([flatten1, flatten2, flatten10, flatten3, flatten4, flatten6,
                                                       flatten7, flatten8, flatten9])
    output = tensorflow.keras.layers.Dense(1, activation='sigmoid', name='output')(final_merge)
    final_model = tensorflow.keras.models.Model(inputs=[input_big, input_small], outputs=[output1, output2, output3,
                                                                                          output4, output6, output7,
                                                                                          output8, output9, output])
    return final_model


tensorflow.keras.backend.set_image_data_format("channels_first")
#test = CreateDenseModel
#model = CreateDenseModel.multi_dense_model(test)

#model.compile(optimizer=SGD(lr=0.001),
#              loss={'malignancy_regression': mse,
#                    'type_classification': categorical_crossentropy},
#              metrics={'malignancy_regression': ['AUC'],
#                       'type_classification': ['categorial_accuracy']})

#print(model.summary())
