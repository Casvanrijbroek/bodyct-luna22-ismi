import tensorflow.keras

l2_lambda = 0.0002
DropP = 0.3

tensorflow.keras.backend.set_image_data_format("channels_first")


def dense_block(conv):
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(conv)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    y = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(x)
    y = tensorflow.keras.layers.BatchNormalization()(y)
    merge = tensorflow.keras.layers.Concatenate(axis=1)([x, y])
    x = tensorflow.keras.layers.Conv3D(filters=12, kernel_size=(3, 3, 3), activation='relu', padding='same',
                                       kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
    x = tensorflow.keras.layers.BatchNormalization()(x)
    merge = tensorflow.keras.layers.Concatenate(axis=1)([merge, x])
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
    merge = tensorflow.keras.layers.Concatenate(axis=1)([x, y])
    for z in range(layers - 2):
        x = tensorflow.keras.layers.Conv3D(filters=filters, kernel_size=(3, 3, 3), activation='relu',
                                           padding='same',
                                           kernel_regularizer=tensorflow.keras.regularizers.l2(l2_lambda))(merge)
        x = tensorflow.keras.layers.BatchNormalization()(x)
        merge = tensorflow.keras.layers.Concatenate(axis=1)([merge, x])
    return x


def multi_dense_model(big_input):
    input_big = tensorflow.keras.layers.Input(shape=big_input)
    prepool = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), padding='same')(input_big)

    pool1 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(dense_block(prepool))
    pool1 = tensorflow.keras.layers.Dropout(DropP)(pool1)
    flatten1 = tensorflow.keras.layers.Flatten()(pool1)

    pool2 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool1, 10, 12))
    pool2 = tensorflow.keras.layers.Dropout(DropP)(pool2)
    flatten2 = tensorflow.keras.layers.Flatten()(pool2)

    pool3 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool2, 20, 12))
    pool3 = tensorflow.keras.layers.Dropout(DropP)(pool3)
    flatten3 = tensorflow.keras.layers.Flatten()(pool3)

    pool4 = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(denser_block(pool3, 20, 24))
    pool4 = tensorflow.keras.layers.Dropout(DropP)(pool4)
    flatten4 = tensorflow.keras.layers.Flatten()(pool4)

    pool5 = denser_block(pool4, 20, 48)
    flatten5 = tensorflow.keras.layers.Flatten()(pool5)

    final_merge = tensorflow.keras.layers.Concatenate(axis=1)([flatten1, flatten2, flatten3, flatten4, flatten5])
    output = tensorflow.keras.layers.Dense(1, activation='sigmoid', name='output')(final_merge)
    output_malignancy = tensorflow.keras.layers.Dense(2, activation='softmax', name='malignancy_regression')(output)
    output_type = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(output)
    final_model = tensorflow.keras.models.Model(inputs=[input_big], outputs=[output_malignancy, output_type])
    return final_model
