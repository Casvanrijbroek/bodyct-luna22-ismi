import tensorflow.keras

tensorflow.keras.backend.set_image_data_format("channels_first")


def classification_layer(inputs):
    x = tensorflow.keras.layers.Dense(512, activation='relu')(inputs)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output_type = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(x)
    return output_type


def regression_layer(inputs):
    x = tensorflow.keras.layers.Dense(512, activation='relu')(inputs)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output_mal = tensorflow.keras.layers.Dense(2, activation='softmax', name='malignancy_regression')(x)
    return output_mal


def add_dense_blocks_old(inputs, filter_size):
    for i in range(3):
        x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                             activation='relu', kernel_initializer='he_normal')(inputs)
        if i != 0:
            inputs = tensorflow.keras.layers.Concatenate(axis=1)([x_1, inputs])  # axis=1
        else:
            inputs = x_1
        x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                             activation='relu', kernel_initializer='he_normal')(inputs)
        inputs = tensorflow.keras.layers.Concatenate(axis=1)([x_2, inputs])  # axis=1
        filter_size += 32
    return inputs


def add_dense_blocks(inputs, filter_size):
    x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                         activation='relu', kernel_initializer='he_normal')(inputs)
    x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                         activation='relu', kernel_initializer='he_normal')(x_1)
    concat_x1 = x_1
    concat_x2 = x_2
    filter_size += 8  # 32
    for i in range(2):
        x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                             activation='relu', kernel_initializer='he_normal',
                                             kernel_regularizer='L2')(concat_x2)
        concat_x1 = tensorflow.keras.layers.Concatenate(axis=1)([x_1, concat_x1])  # axis=1
        x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                             activation='relu', kernel_initializer='he_normal',
                                             kernel_regularizer='L2')(concat_x1)
        concat_x2 = tensorflow.keras.layers.Concatenate(axis=1)([x_2, concat_x2])  # axis=1
        filter_size += 16  # 32
    return x_2


def add_transition_blocks(x, transition_filter):
    x = tensorflow.keras.layers.Conv3D(filters=transition_filter, kernel_size=(1, 1, 1), padding='same',
                                       activation='relu', kernel_initializer='he_normal')(x)
    x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    return x


def add_network_blocks(x, block_amount, filter_list):
    # Create dense block
    for block in range(block_amount):
        # Add dense block
        x = add_dense_blocks(x, filter_list[block])
        # Create transition layer
        x = add_transition_blocks(x, filter_list[block]//2)
    return x


def build_model(input_shape, filter_list=[160, 176, 184, 188, 190]):
    input_layer = tensorflow.keras.layers.Input(shape=input_shape)
    x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                                       activation='relu', kernel_initializer='he_normal')(input_layer)
    x = add_network_blocks(x, 4, filter_list)
    x = tensorflow.keras.layers.MaxPooling3D()(x)

    x = add_dense_blocks(x, filter_list[-1])

    x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(x)
    x = tensorflow.keras.layers.Flatten(name='bifurcation_layer')(x)
    output_malignancy = regression_layer(x)
    output_type = classification_layer(x)
    d_model = tensorflow.keras.Model(inputs=input_layer, outputs=[output_malignancy, output_type])
    return d_model
