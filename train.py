import random
from pathlib import Path
from typing import Tuple

import numpy.random
from scipy.ndimage import rotate
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

from balanced_sampler import sample_balanced, UndersamplingIterator
from helper_functions import MLProblem
from data import load_dataset
import resnet_3d
from custom_callbacks import DualBestSave, DualEarlyStopping
# autograph.set_verbosity(2)

# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")


# This should point at the directory containing the source LUNA22 prequel dataset
DATA_DIRECTORY = Path("/home/lbosch/data/LUNA22 prequel")

# This should point at a directory to put the preprocessed/generated datasets from the source data
GENERATED_DATA_DIRECTORY = Path().absolute()

# This should point at a directory to store the training output files
TRAINING_OUTPUT_DIRECTORY = Path().absolute()


# Load dataset
# This method will generate a preprocessed dataset from the source data if it is not present (only needs to be done once)
# Otherwise it will quickly load the generated dataset from disk
full_dataset = load_dataset(
    input_size=64,
    new_spacing_mm=1.0,
    cross_slices_only=False,
    generate_if_not_present=True,
    always_generate=False,
    source_data_dir=DATA_DIRECTORY,
    generated_data_dir=GENERATED_DATA_DIRECTORY,
)
inputs = full_dataset["inputs"]


# Here you can switch the machine learning problem to solve
problem = MLProblem.nodule_type_prediction

# Configure problem specific parameters
if problem == MLProblem.malignancy_prediction:
    # We made this problem a binary classification problem:
    # 0 - benign, 1 - malignant
    num_classes = 2
    batch_size = 32
    # Take approx. 15% of all samples for the validation set and ensure it is a multiple of the batch size
    num_validation_samples = int(len(inputs) * 0.15 / batch_size) * batch_size
    labels = full_dataset["labels_malignancy"]
    # It is possible to generate training labels yourself using the raw annotations of the radiologists...
    labels_raw = full_dataset["labels_malignancy_raw"]
elif problem == MLProblem.nodule_type_prediction:
    # We made this problem a multiclass classification problem with three classes:
    # 0 - non-solid, 1 - part-solid, 2 - solid
    num_classes = 3
    batch_size = 30  # make this a factor of three to fit three classes evenly per batch during training
    # This dataset has only few part-solid nodules in the dataset, so we make a tiny validation set
    num_validation_samples = batch_size * 2
    labels = full_dataset["labels_nodule_type"]
    # It is possible to generate training labels yourself using the raw annotations of the radiologists...
    labels_raw = full_dataset["labels_nodule_type_raw"]
else:
    raise NotImplementedError(f"An unknown MLProblem was specified: {problem}")

print(
    f"Finished loading data for MLProblem: {problem}... X:{inputs.shape} Y:{labels.shape}"
)

# partition small and class balanced validation set from all data
validation_indices = sample_balanced(
    input_labels=np.argmax(labels, axis=1),
    required_samples=num_validation_samples,
    class_balance=None,  # By default sample with equal probability, e.g. for two classes : {0: 0.5, 1: 0.5}
    shuffle=True,
)

validation_mask = np.isin(np.arange(len(labels)), list(validation_indices.values()))

labels_malignancy = full_dataset["labels_malignancy"]
labels_type = full_dataset['labels_nodule_type']

training_inputs = inputs[~validation_mask, :]
training_labels_malignancy = labels_malignancy[~validation_mask, :]
training_labels_type = labels_type[~validation_mask, :]
validation_inputs = inputs[validation_mask, :]
validation_labels_malignancy = labels_malignancy[validation_mask, :]
validation_labels_type = labels_type[validation_mask, :]

print(f"Splitted data into training and validation sets:")
training_class_counts = np.unique(
    np.argmax(training_labels_malignancy, axis=1), return_counts=True
)[1]
validation_class_counts = np.unique(
    np.argmax(validation_labels_malignancy, axis=1), return_counts=True
)[1]
print(
    f"Training   set: {training_inputs.shape} {training_labels_malignancy.shape} {training_class_counts}"
)
print(
    f"Validation set: {validation_inputs.shape} {validation_labels_malignancy.shape} {validation_class_counts}"
)


# Split dataset into two data generators for training and validation
# Technically we could directly pass the data into the fit function of the model
# But using generators allows for a simple way to add augmentations and preprocessing
# It also allows us to balance the batches per class using undersampling

# The following methods can be used to implement custom preprocessing/augmentation during training




def clip_and_scale(
    data: np.ndarray, min_value: float = -1000.0, max_value: float = 400.0
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


def random_flip_augmentation(
    input_sample: np.ndarray, axis: Tuple[int, ...] = (1, 2)
) -> np.ndarray:
    for ax in axis:
        if np.random.random_sample() > 0.5:
            input_sample = np.flip(input_sample, axis=ax)
    return input_sample


def shared_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    """Preprocessing that is used by both the training and validation sets during training

    :param input_batch: np.ndarray [batch_size x channels x dim_x x dim_y]
    :return: np.ndarray preprocessed batch
    """
    input_batch = clip_and_scale(input_batch, min_value=-1000.0, max_value=400.0)
    # Can add more preprocessing here...
    return input_batch


def random_noise(data: np.ndarray, stdev: float = 0.001) -> np.ndarray:
    """ Introduces random noise generated by a gaussian with a given standard deviation

    :param data:
    :param stdev:
    :return:
    """
    noise = np.random.normal(0, stdev, data.shape)
    new_data = data + noise
    return new_data


def random_brightness(data: np.ndarray, factor: float) -> np.ndarray:
    random_value = (np.random.random_sample()*(factor*2))-(factor)
    factor = (1-float(random_value))
    return_values = np.clip(0.5 + factor * data - factor * 0.5, 0, 1)
    return return_values


def random_rotate(data: np.ndarray, rotation: float) -> np.ndarray:

    # random_rotation = [(1, 2), (2, 3), (1, 3)][np.random.randint(1, 3)]
    random_rotation = (1,2)
    angle = ((numpy.random.random_sample()*(rotation*2))-rotation)
    new_image = rotate(data[:, :, :, :], angle=angle*360, reshape=False, mode="nearest", axes=random_rotation)

    return new_image

def alt_random_rotate(data: np.ndarray):
    data = np.rot90(data, k=np.random.randint(0, 3), axes=(1, 2))
    data = np.rot90(data, k=np.random.randint(0, 3), axes=(1, 3))
    data = np.rot90(data, k=np.random.randint(0, 3), axes=(3, 2))
    return data

def train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)

    output_batch = []

    for sample in input_batch:
        # plt.imshow(sample[0, :, :, 32])
        # plt.show()
        sample = random_flip_augmentation(sample, axis=(1, 2))
        # sample = random_rotate(sample, 0.2)
        sample = random_noise(sample, 0.004)
        sample = random_brightness(sample, 0.35)
        sample = alt_random_rotate(sample)
        # plt.imshow(sample[0, :, :, 32])
        # plt.show()
        output_batch.append(sample)

    return np.array(output_batch)


def validation_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)
    return input_batch


training_data_generator = UndersamplingIterator(
    training_inputs,
    labels_malignancy=training_labels_malignancy,
    labels_type=training_labels_type,
    problem=problem,
    shuffle=True,
    preprocess_fn=train_preprocess_fn,
    batch_size=batch_size,
)
validation_data_generator = UndersamplingIterator(
    validation_inputs,
    labels_malignancy=validation_labels_malignancy,
    labels_type=validation_labels_type,
    problem=problem,
    shuffle=False,
    preprocess_fn=validation_preprocess_fn,
    batch_size=batch_size,
)

malignancy_classes = 1  # Actually 2, but goal is to find value between 0 and 1
type_classes = 3        # Solid, partly-solid, non-solid
# model = densenet.dense_model(malignancy_classes, type_classes)

model = resnet_3d.CustomResnet3DBuilder.build_resnet_18((1, 64, 64, 64), 3)


model.compile(optimizer=Adam(learning_rate=0.001),
              loss={'malignancy_regression': losses.mse,
                    'type_classification': losses.categorical_crossentropy},
              metrics={'malignancy_regression': ['AUC'],
                       'type_classification': ['categorical_accuracy']})
# Show the model layers
print(model.summary())

# Start actual training process
output_model_file = (
    TRAINING_OUTPUT_DIRECTORY / f"resnet_{problem.value}_best_type_val_accuracy.h5"
)

callbacks = [
    TerminateOnNaN(),
    DualBestSave(
        str(output_model_file),
        monitor="val_type_classification_categorical_accuracy",
        monitor2="val_malignancy_regression_auc",
        mode="auto",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq="epoch",
    ),
    DualEarlyStopping(
        monitor="val_type_classification_categorical_accuracy",
        monitor2="val_malignancy_regression_auc",
        mode="auto",
        min_delta=0,
        patience=100,
        verbose=1,
    ),
]
history = model.fit(
    training_data_generator,
    steps_per_epoch=len(training_data_generator),
    validation_data=validation_data_generator,
    validation_steps=None,
    validation_freq=1,
    epochs=250,
    callbacks=callbacks,
    verbose=2,
)


# generate a plot using the training history...
output_history_img_file_type = (
    TRAINING_OUTPUT_DIRECTORY / f"dense_type_classification_train_plot.png"
)
output_history_img_file_mal = (
    TRAINING_OUTPUT_DIRECTORY / f"dense_malignancy_prediction_train_plot.png"
)

print(f"Saving training plots to: {TRAINING_OUTPUT_DIRECTORY}")
print(history.history.keys())
# Possible values: dict_keys(['loss', 'malignancy_regression_loss', 'type_classification_loss', 'malignancy_regression_auc', 'type_classification_categorical_accuracy', 'val_loss', 'val_malignancy_regression_loss', 'val_type_classification_loss', 'val_malignancy_regression_auc', 'val_type_classification_categorical_accuracy'])
plt.plot(history.history["type_classification_categorical_accuracy"])
plt.plot(history.history["val_type_classification_categorical_accuracy"])
plt.plot(history.history["type_classification_loss"])
plt.plot(history.history["val_type_classification_loss"])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
plt.savefig(str(output_history_img_file_type), bbox_inches="tight")
plt.clf()
plt.plot(history.history["malignancy_regression_auc"])
plt.plot(history.history["val_malignancy_regression_auc"])
plt.plot(history.history["malignancy_regression_loss"])
plt.plot(history.history["val_malignancy_regression_loss"])
plt.title("model AUC")
plt.ylabel("AUC")
plt.xlabel("Epoch")
plt.legend(["AUC", "Validation AUC", "Loss", "Validation Loss"])
plt.savefig(str(output_history_img_file_mal), bbox_inches="tight")
