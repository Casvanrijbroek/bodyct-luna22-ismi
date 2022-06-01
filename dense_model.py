import tensorflow.keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy, mse
from tensorflow.keras.utils import to_categorical


class CreateModel:
    def __init__(self):
        pass

    @staticmethod
    def classification_layer(inputs):
        x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(inputs)
        x = tensorflow.keras.layers.Flatten()(x)
        x = tensorflow.keras.layers.Dense(512, activation='relu')(x)
        x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
        output_type = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(x)
        return output_type

    @staticmethod
    def regression_layer(inputs):
        x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(inputs)
        x = tensorflow.keras.layers.Flatten()(x)
        x = tensorflow.keras.layers.Dense(512, activation='relu')(x)
        x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
        output_mal = tensorflow.keras.layers.Dense(1, activation='sigmoid', name='malignancy_regression')(x)
        return output_mal

    @staticmethod
    def add_dense_blocks(inputs, filter_size):
        for i in range(3):
            x = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same')(inputs)
            inputs = tensorflow.keras.layers.Concatenate()([x, inputs])
            x = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same')(inputs)
            inputs = tensorflow.keras.layers.Concatenate()([x, inputs])
            filter_size += 32
        return inputs

    @staticmethod
    def add_transition_blocks(x, idx):
        if idx <= 1:
            x = tensorflow.keras.layers.Conv3D(filters=80, kernel_size=(1, 1, 1), padding='same')(x)
            x = tensorflow.keras.layers.MaxPooling3D()(x)
        elif idx == 2:
            x = tensorflow.keras.layers.Conv3D(filters=96, kernel_size=(1, 1, 1), padding='same')(x)
            x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        elif idx == 3:
            x = tensorflow.keras.layers.Conv3D(filters=94, kernel_size=(1, 1, 1), padding='same')(x)
            x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        return x

    def add_network_blocks(self, x):
        filter_list = [160, 176, 184, 188, 190]
        # Create dense block
        for block, filter_size in enumerate(filter_list):
            # Add dense block
            x = self.add_dense_blocks(x, filter_size)
            # Create transition layer
            x = self.add_transition_blocks(x, block)
        return x

    def dense_model(self):
        input_layer = tensorflow.keras.layers.Input(shape=(64, 64, 64, 1))
        x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(input_layer)
        x = self.add_network_blocks(self, x)
        x = tensorflow.keras.layers.MaxPooling3D()(x)

        output_malignancy = self.regression_layer(x)
        output_type = self.classification_layer(x)
        d_model = tensorflow.keras.Model(inputs=input_layer, outputs=[output_malignancy, output_type])
        return d_model


test = CreateModel
model = CreateModel.dense_model(test)

model.compile(optimizer=SGD(lr=0.001),
              loss={'malignancy_regression': mse,
                    'type_classification': categorical_crossentropy},
              metrics={'malignancy_regression': ['AUC'],
                       'type_classification': ['categorical_accuracy']})

# Show the model layers
print(model.summary())



