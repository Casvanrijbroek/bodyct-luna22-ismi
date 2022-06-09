# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-import-not-at-top
# pylint: disable=g-classes-have-attributes
"""Callbacks: utilities called at certain points during model training."""

import collections
import copy
import csv
import json
import os
import re
import sys
import time


from keras import backend
from keras.distribute import distributed_file_utils
from keras.distribute import worker_training_state
from keras.optimizers.schedules import learning_rate_schedule
from keras.utils import generic_utils
from keras.utils import io_utils
from keras.utils import tf_utils
from keras.utils import version_utils
from keras.utils.data_utils import Sequence
from keras.utils.generic_utils import Progbar
from keras.utils.mode_keys import ModeKeys
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

try:
    import requests
except ImportError:
    requests = None


# Note: `configure_callbacks` is only used in TF1.
def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode=ModeKeys.TRAIN):
    """Configures callbacks for use in various training loops.

    Args:
        callbacks: List of Callbacks.
        model: Model being trained.
        do_validation: Whether or not validation loop will be run.
        batch_size: Number of samples per batch.
        epochs: Number of epoch to train.
        steps_per_epoch: Number of batches to run per training epoch.
        samples: Number of training samples.
        verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
        count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
        mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
          Which loop mode to configure callbacks for.

    Returns:
        Instance of CallbackList used to control all Callbacks.
    """
    # Check if callbacks have already been configured.
    if isinstance(callbacks, CallbackList):
        return callbacks

    if not callbacks:
        callbacks = []

    # Add additional callbacks during training.
    if mode == ModeKeys.TRAIN:
        model.history = History()
        callbacks = [BaseLogger()] + (callbacks or []) + [model.history]
        if verbose:
            callbacks.append(ProgbarLogger(count_mode))
    callback_list = CallbackList(callbacks)

    # Set callback model
    callback_model = model._get_callback_model()  # pylint: disable=protected-access
    callback_list.set_model(callback_model)

    set_callback_parameters(
        callback_list,
        model,
        do_validation=do_validation,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        samples=samples,
        verbose=verbose,
        mode=mode)

    callback_list.model.stop_training = False
    return callback_list


def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
    """Sets callback parameters.

    Args:
        callback_list: CallbackList instance.
        model: Model being trained.
        do_validation: Whether or not validation loop will be run.
        batch_size: Number of samples per batch.
        epochs: Number of epoch to train.
        steps_per_epoch: Number of batches to run per training epoch.
        samples: Number of training samples.
        verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
        mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
          Which loop mode to configure callbacks for.
    """
    metric_names = model.metrics_names
    for cbk in callback_list:
        if isinstance(cbk, (BaseLogger, ProgbarLogger)):
            cbk.stateful_metrics = metric_names[1:]  # Exclude `loss`

    # Set callback parameters
    callback_metrics = []
    # When we have deferred build scenario with iterator input, we will compile
    # when we standardize first batch of data.
    if mode != ModeKeys.PREDICT:
        callback_metrics = copy.copy(metric_names)
        if do_validation:
            callback_metrics += ['val_' + n for n in metric_names]
    callback_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': samples,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    }
    callback_list.set_params(callback_params)


def _is_generator_like(data):
    """Checks if data is a generator, Sequence, or Iterator."""
    return (hasattr(data, '__next__') or hasattr(data, 'next') or isinstance(
        data, (Sequence, tf.compat.v1.data.Iterator, tf.data.Iterator)))


def make_logs(model, logs, outputs, mode, prefix=''):
    """Computes logs for sending to `on_batch_end` methods."""
    metric_names = model.metrics_names
    if mode in {ModeKeys.TRAIN, ModeKeys.TEST} and metric_names:
        for label, output in zip(metric_names, outputs):
            logs[prefix + label] = output
    else:
        logs['outputs'] = outputs
    return logs


@keras_export('keras.callbacks.CallbackList')
class CallbackList:
    """Container abstracting a list of callbacks."""

    def __init__(self,
                 callbacks=None,
                 add_history=False,
                 add_progbar=False,
                 model=None,
                 **params):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
          callbacks: List of `Callback` instances.
          add_history: Whether a `History` callback should be added, if one does not
            already exist in the `callbacks` list.
          add_progbar: Whether a `ProgbarLogger` callback should be added, if one
            does not already exist in the `callbacks` list.
          model: The `Model` these callbacks are used with.
          **params: If provided, parameters will be passed to each `Callback` via
            `Callback.set_params`.
        """
        self.callbacks = tf.nest.flatten(callbacks) if callbacks else []
        self._add_default_callbacks(add_history, add_progbar)

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

        # Performance optimization: determines if batch hooks need to be called.
        # pylint: disable=protected-access
        self._supports_tf_logs = all(
            getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks)
        self._batch_hooks_support_tf_logs = all(
            getattr(cb, '_supports_tf_logs', False)
            for cb in self.callbacks
            if cb._implements_train_batch_hooks() or cb
                ._implements_test_batch_hooks() or cb._implements_predict_batch_hooks())

        self._should_call_train_batch_hooks = any(
            cb._implements_train_batch_hooks() for cb in self.callbacks)
        self._should_call_test_batch_hooks = any(
            cb._implements_test_batch_hooks() for cb in self.callbacks)
        self._should_call_predict_batch_hooks = any(
            cb._implements_predict_batch_hooks() for cb in self.callbacks)
        # pylint: enable=protected-access

        self._disallow_batch_hooks_in_ps_strategy()

        # Performance check: Check batch hooks for slowness compared to batch time.
        # Only run check for custom callbacks (i.e. not present in this file).
        self._check_timing = any(
            cbk.__class__.__name__ not in globals() for cbk in self.callbacks)
        self._num_batches_for_timing_check = 5
        self._hook_times = {}
        self._batch_start_time = None
        self._batch_times = []

    def _add_default_callbacks(self, add_history, add_progbar):
        """Adds `Callback`s that are always present."""
        self._progbar = None
        self._history = None

        for cb in self.callbacks:
            if isinstance(cb, ProgbarLogger):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

        if self._progbar is None and add_progbar:
            self._progbar = ProgbarLogger(count_mode='steps')
            self.callbacks.append(self._progbar)

    def _process_logs(self, logs, is_batch_hook=False):
        """Turns tensors into numpy arrays or Python scalars if necessary."""
        if logs is None:
            return {}
        if self._supports_tf_logs:
            return logs
        if is_batch_hook and self._batch_hooks_support_tf_logs:
            return logs
        return tf_utils.sync_to_numpy_or_python_type(logs)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return

        if hook == 'begin':
            self._call_batch_begin_hook(mode, batch, logs)
        elif hook == 'end':
            self._call_batch_end_hook(mode, batch, logs)
        else:
            raise ValueError(
                f'Unrecognized hook: {hook}. Expected values are ["begin", "end"]')

    def _call_batch_begin_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_begin` methods."""
        hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
        self._call_batch_hook_helper(hook_name, batch, logs)

        if self._check_timing:
            self._batch_start_time = time.time()

    def _call_batch_end_hook(self, mode, batch, logs):
        """Helper function for `on_*_batch_end` methods."""
        hook_name = 'on_{mode}_batch_end'.format(mode=mode)

        if self._check_timing and batch >= 1:
            batch_time = time.time() - self._batch_start_time
            self._batch_times.append(batch_time)

        self._call_batch_hook_helper(hook_name, batch, logs)

        if len(self._batch_times) >= self._num_batches_for_timing_check:
            end_hook_name = hook_name
            begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
            avg_batch_time = sum(self._batch_times) / len(self._batch_times)
            avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(
                self._hook_times[end_hook_name])
            avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(
                self._hook_times[begin_hook_name])

            threshold_time = 1.0 * avg_batch_time
            warning_msg = ('Callback method `{hook}` is slow compared to '
                           'the batch time (batch time: {batch_time:.4f}s vs '
                           '`{hook}` time: {hook_time:.4f}s). Check your callbacks.')
            if avg_begin_hook_time > threshold_time:
                logging.warning(warning_msg.format(
                    hook=begin_hook_name,
                    batch_time=avg_batch_time,
                    hook_time=avg_begin_hook_time))
            if avg_end_hook_time > threshold_time:
                logging.warning(warning_msg.format(
                    hook=end_hook_name,
                    batch_time=avg_batch_time,
                    hook_time=avg_end_hook_time))
            self._check_timing = False
            self._batch_start_time = None
            self._batch_times = []
            self._hook_times = {}

    def _call_batch_hook_helper(self, hook_name, batch, logs):
        """Helper function for `on_*_batch_*` methods."""
        if self._check_timing:
            start_time = time.time()

        logs = self._process_logs(logs, is_batch_hook=True)
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)

        if self._check_timing:
            if hook_name not in self._hook_times:
                self._hook_times[hook_name] = []
            self._hook_times[hook_name].append(time.time() - start_time)

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_begin()
        elif mode == ModeKeys.TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == ModeKeys.TRAIN:
            self.on_train_end()
        elif mode == ModeKeys.TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    def on_batch_begin(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.

        This function should only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.train_step`. Typically,
              the values of the `Model`'s metrics are returned.  Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        if self._should_call_train_batch_hooks:
            self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.test_step`. Typically,
              the values of the `Model`'s metrics are returned.  Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        if self._should_call_test_batch_hooks:
            self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        if self._should_call_test_batch_hooks:
            self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict, contains the return value of `model.predict_step`,
              it typically returns a dict with a key 'outputs' containing
              the model's outputs.
        """
        if self._should_call_predict_batch_hooks:
            self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        if self._should_call_predict_batch_hooks:
            self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently, no data is passed via this argument
              for this method, but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.

        Args:
            logs: Dict. Currently, no data is passed via this argument
              for this method, but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.

        Args:
            logs: Dict. Currently, no data is passed via this argument
              for this method, but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs=None):
        """Calls the 'on_predict_begin` methods of its callbacks.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.

        Args:
            logs: Dict. Currently, no data is passed via this argument
              for this method, but that may change in the future.
        """
        logs = self._process_logs(logs)
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)

    def _disallow_batch_hooks_in_ps_strategy(self):
        """Error out if batch-level callbacks are passed with PSStrategy."""
        # pylint: disable=protected-access
        strategy = tf.distribute.get_strategy()
        if strategy._should_use_with_coordinator:
            unsupported_callbacks = []
            for cb in self.callbacks:
                # These Callbacks can accept RemoteValues directly.
                if getattr(cb, '_supports_tf_logs', False):
                    continue
                if (cb._implements_train_batch_hooks() or
                        cb._implements_test_batch_hooks() or
                        cb._implements_predict_batch_hooks()):
                    unsupported_callbacks.append(cb)
            if unsupported_callbacks:
                raise ValueError(
                    'Batch-level `Callback`s are not supported with '
                    '`ParameterServerStrategy`. Found unsupported '
                    f'callbacks: {unsupported_callbacks}')
        # pylint: enable=protected-access


@keras_export('keras.callbacks.Callback')
class Callback:
    """Abstract base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
    `predict` in order to hook into the various stages of the model training and
    inference lifecycle.

    To create a custom callback, subclass `keras.callbacks.Callback` and override
    the method associated with the stage of interest. See
    https://www.tensorflow.org/guide/keras/custom_callback for more information.

    Example:

    >>> training_finished = False
    >>> class MyCallback(tf.keras.callbacks.Callback):
    ...   def on_train_end(self, logs=None):
    ...     global training_finished
    ...     training_finished = True
    >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    >>> model.compile(loss='mean_squared_error')
    >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
    ...           callbacks=[MyCallback()])
    >>> assert training_finished == True

    If you want to use `Callback` objects in a custom training loop:

    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop. Like this:

       ```
       callbacks =  tf.keras.callbacks.CallbackList([...])
       callbacks.append(...)

       callbacks.on_train_begin(...)
       for epoch in range(EPOCHS):
         callbacks.on_epoch_begin(epoch)
         for i, data in dataset.enumerate():
           callbacks.on_train_batch_begin(i)
           batch_logs = model.train_step(data)
           callbacks.on_train_batch_end(i, batch_logs)
         epoch_logs = ...
         callbacks.on_epoch_end(epoch, epoch_logs)
       final_logs=...
       callbacks.on_train_end(final_logs)
       ```

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """

    def __init__(self):
        self.validation_data = None  # pylint: disable=g-missing-from-attributes
        self.model = None
        # Whether this Callback should only run on the chief worker in a
        # Multi-Worker setting.
        # TODO(omalleyt): Make this attr public once solution is stable.
        self._chief_worker_only = None
        self._supports_tf_logs = False

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    @doc_controls.for_subclass_implementers
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should only
        be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result keys
              are prefixed with `val_`. For training epoch, the values of the
             `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
               0.7}`.
        """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    @generic_utils.default
    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N`
        batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    @doc_controls.for_subclass_implementers
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to `on_epoch_end()`
              is passed to this argument for this method but that may change in
              the future.
        """

    @doc_controls.for_subclass_implementers
    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    @doc_controls.for_subclass_implementers
    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this method
              but that may change in the future.
        """

    def _implements_train_batch_hooks(self):
        """Determines if this Callback should be called for each train batch."""
        return (not generic_utils.is_default(self.on_batch_begin) or
                not generic_utils.is_default(self.on_batch_end) or
                not generic_utils.is_default(self.on_train_batch_begin) or
                not generic_utils.is_default(self.on_train_batch_end))

    def _implements_test_batch_hooks(self):
        """Determines if this Callback should be called for each test batch."""
        return (not generic_utils.is_default(self.on_test_batch_begin) or
                not generic_utils.is_default(self.on_test_batch_end))

    def _implements_predict_batch_hooks(self):
        """Determines if this Callback should be called for each predict batch."""
        return (not generic_utils.is_default(self.on_predict_batch_begin) or
                not generic_utils.is_default(self.on_predict_batch_end))


@keras_export('keras.callbacks.DualBestSave')
class DualBestSave(Callback):
    """Callback to save the Keras model or model weights at some frequency.

    `ModelCheckpoint` callback is used in conjunction with training using
    `model.fit()` to save a model or weights (in a checkpoint file) at some
    interval, so the model or weights can be loaded later to continue the training
    from the state saved.

    A few options this callback provides include:

    - Whether to only keep the model that has achieved the "best performance" so
      far, or whether to save the model at the end of every epoch regardless of
      performance.
    - Definition of 'best'; which quantity to monitor and whether it should be
      maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving at
      the end of every epoch, or after a fixed number of training batches.
    - Whether only weights are saved, or the whole model is saved.

    Note: If you get `WARNING:tensorflow:Can save best model only with <name>
    available, skipping` see the description of the `monitor` argument for
    details on how to get this right.

    Example:

    ```python
    model.compile(loss=..., optimizer=...,
                  metrics=['accuracy'])

    EPOCHS = 10
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(checkpoint_filepath)
    ```

    Args:
        filepath: string or `PathLike`, path to save the model file. e.g.
          filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
          can contain named formatting options, which will be filled the value of
          `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
          `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
          checkpoints will be saved with the epoch number and the validation loss
          in the filename. The directory of the filepath should not be reused by
          any other callbacks to avoid conflicts.
        monitor: The metric name to monitor. Typically the metrics are set by the
          `Model.compile` method. Note:

          * Prefix the name with `"val_`" to monitor validation metrics.
          * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
          * If you specify metrics as strings, like `"accuracy"`, pass the same
            string (with or without the `"val_"` prefix).
          * If you pass `metrics.Metric` objects, `monitor` should be set to
            `metric.name`
          * If you're not sure about the metric names you can check the contents
            of the `history.history` dictionary returned by
            `history = model.fit()`
          * Multi-output models set additional prefixes on the metric names.

        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
          displays messages when the callback takes an action.
        save_best_only: if `save_best_only=True`, it only saves when the model
          is considered the "best" and the latest best model according to the
          quantity monitored will not be overwritten. If `filepath` doesn't
          contain formatting options like `{epoch}` then `filepath` will be
          overwritten by each new better model.
        mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
          decision to overwrite the current save file is made based on either
          the maximization or the minimization of the monitored quantity.
          For `val_acc`, this should be `max`, for `val_loss` this should be
          `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
          monitored are 'acc' or start with 'fmeasure' and are set to `min` for
          the rest of the quantities.
        save_weights_only: if True, then only the model's weights will be saved
          (`model.save_weights(filepath)`), else the full model is saved
          (`model.save(filepath)`).
        save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
          the model after each epoch. When using integer, the callback saves the
          model at end of this many batches. If the `Model` is compiled with
          `steps_per_execution=N`, then the saving criteria will be
          checked every Nth batch. Note that if the saving isn't aligned to
          epochs, the monitored metric may potentially be less reliable (it
          could reflect as little as 1 batch, since the metrics get reset every
          epoch). Defaults to `'epoch'`.
        options: Optional `tf.train.CheckpointOptions` object if
          `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
          object if `save_weights_only` is false.
        initial_value_threshold: Floating point initial "best" value of the metric
          to be monitored. Only applies if `save_best_value=True`. Only overwrites
          the model weights already saved if the performance of current
          model is better than this value.
        **kwargs: Additional arguments for backwards compatibility. Possible key
          is `period`.
    """

    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 monitor2="val_loss",
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 options=None,
                 initial_value_threshold=None,
                 **kwargs):
        super(DualBestSave, self).__init__()
        self._supports_tf_logs = True
        self.monitor = monitor
        self.monitor2 = monitor2
        self.verbose = verbose
        self.filepath = io_utils.path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold

        if save_weights_only:
            if options is None or isinstance(
                    options, tf.train.CheckpointOptions):
                self._options = options or tf.train.CheckpointOptions()
            else:
                raise TypeError(
                    'If save_weights_only is True, then `options` must be '
                    f'either None or a tf.train.CheckpointOptions. Got {options}.')
        else:
            if options is None or isinstance(options, tf.saved_model.SaveOptions):
                self._options = options or tf.saved_model.SaveOptions()
            else:
                raise TypeError(
                    'If save_weights_only is False, then `options` must be '
                    f'either None or a tf.saved_model.SaveOptions. Got {options}.')

        # Deprecated field `load_weights_on_restart` is for loading the checkpoint
        # file from `filepath` at the start of `model.fit()`
        # TODO(rchao): Remove the arg during next breaking release.
        if 'load_weights_on_restart' in kwargs:
            self.load_weights_on_restart = kwargs['load_weights_on_restart']
            logging.warning('`load_weights_on_restart` argument is deprecated. '
                            'Please use `model.load_weights()` for loading weights '
                            'before the start of `model.fit()`.')
        else:
            self.load_weights_on_restart = False

        # Deprecated field `period` is for the number of epochs between which
        # the model is saved.
        if 'period' in kwargs:
            self.period = kwargs['period']
            logging.warning('`period` argument is deprecated. Please use `save_freq` '
                            'to specify the frequency in number of batches seen.')
        else:
            self.period = 1

        if mode not in ['auto', 'min', 'max']:
            logging.warning('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf

        if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
            raise ValueError(
                f'Unrecognized save_freq: {self.save_freq}. '
                'Expected save_freq are "epoch" or integer')

        # Only the chief worker writes model checkpoints, but all workers
        # restore checkpoint at on_train_begin().
        self._chief_worker_only = False

    def on_train_begin(self, logs=None):
        if self.load_weights_on_restart:
            filepath_to_load = (
                self._get_most_recently_modified_file_matching_pattern(self.filepath))
            if (filepath_to_load is not None and
                    self._checkpoint_exists(filepath_to_load)):
                try:
                    # `filepath` may contain placeholders such as `{epoch:02d}`, and
                    # thus it attempts to load the most recently modified file with file
                    # name matching the pattern.
                    self.model.load_weights(filepath_to_load)
                except (IOError, ValueError) as e:
                    raise ValueError(
                        f'Error loading file from {filepath_to_load}. Reason: {e}')

    def _implements_train_batch_hooks(self):
        # Only call batch hooks when saving on batch
        return self.save_freq != 'epoch'

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == 'epoch':
            return False

        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.sync_to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)

            try:
                if self.save_best_only:
                    type_acc = logs.get(self.monitor)
                    mal_auc = logs.get(self.monitor2)

                    current = (mal_auc * 0.75) + (type_acc * 0.25)


                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: {self.monitor} and {self.monitor2} improved '
                                    f'from {self.best:.5f} to {current:.5f}, '
                                    f'saving model to {filepath}')
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                io_utils.print_msg(
                                    f'\nEpoch {epoch + 1}: '
                                    f'{self.monitor} and {self.monitor2} did not improve from {self.best:.5f}')
                else:
                    if self.verbose > 0:
                        io_utils.print_msg(
                            f'\nEpoch {epoch + 1}: saving model to {filepath}')
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)

                self._maybe_remove_file()
            except IsADirectoryError as e:  # h5py 3.x
                raise IOError('Please specify a non-directory filepath for '
                              'ModelCheckpoint. Filepath used is an existing '
                              f'directory: {filepath}')
            except IOError as e:  # h5py 2.x
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  f'directory: f{filepath}')
                # Re-throw the error for any other causes.
                raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}`,`{batch:02d}`
            # and `{mape:.2f}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            if batch is None or 'batch' in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f'Reason: {e}')
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath

    def _maybe_remove_file(self):
        # Remove the checkpoint directory in multi-worker training where this worker
        # should not checkpoint. It is a dummy directory previously saved for sync
        # distributed training.
        distributed_file_utils.remove_temp_dir_with_filepath(
            self._write_filepath, self.model.distribute_strategy)

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        if filepath.endswith('.h5'):
            return tf.io.gfile.exists(filepath)
        tf_saved_model_exists = tf.io.gfile.exists(filepath)
        tf_weights_only_checkpoint_exists = tf.io.gfile.exists(
            filepath + '.index')
        return tf_saved_model_exists or tf_weights_only_checkpoint_exists

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.

        Pattern may contain python formatting placeholder. If
        `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
        check for most recently modified one that matches the pattern.

        In the rare case where there are more than one pattern-matching file having
        the same modified time that is most recent among all, return the filepath
        that is largest (by `>` operator, lexicographically using the numeric
        equivalents). This provides a tie-breaker when multiple files are most
        recent. Note that a larger `filepath` can sometimes indicate a later time of
        modification (for instance, when epoch/batch is used as formatting option),
        but not necessarily (when accuracy or loss is used). The tie-breaker is
        put in the logic as best effort to return the most recent, and to avoid
        undeterministic result.

        Modified time of a file is obtained with `os.path.getmtime()`.

        This utility function is best demonstrated via an example:

        ```python
        file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name) for file_name in
            ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
        ]
        for file_path in file_paths:
          # Write something to each of the files
        self.assertEqual(
            _get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1])
        ```

        Args:
            pattern: The file pattern that may optionally contain python placeholder
                such as `{epoch:02d}`.

        Returns:
            The most recently modified file's full filepath matching `pattern`. If
            `pattern` does not contain any placeholder, this returns the filepath
            that
            exactly matches `pattern`. Returns `None` if no match is found.
        """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

        # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
        # use that as it is more robust than `os.path.getmtime()`.
        latest_tf_checkpoint = tf.train.latest_checkpoint(dir_name)
        if latest_tf_checkpoint is not None and re.match(
                base_name_regex, os.path.basename(latest_tf_checkpoint)):
            return latest_tf_checkpoint

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if tf.io.gfile.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (file_path_with_largest_file_name is None or
                            file_path > file_path_with_largest_file_name):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found, reset
                        # the counter for the number of files with latest modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the most recent,
                        # increment the counter for the number of files with latest modified
                        # time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time, return
            # the file path with the largest file name.
            return file_path_with_largest_file_name


@keras_export('keras.callbacks.DualEarlyStopping')
class DualEarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.
    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.
    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.
    Args:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
          displays messages when the callback takes an action.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used. An epoch will be restored regardless
          of the performance relative to the `baseline`. If no epoch
          improves on `baseline`, training will run for `patience`
          epochs and restore weights from the best epoch in that set.
    Example:
    >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4
    """

    def __init__(self,
                 monitor='val_loss',
                 monitor2="val_los",
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(DualEarlyStopping, self).__init__()

        self.monitor = monitor
        self.monitor2 = monitor2
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
                    self.monitor.endswith('auc')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        'Restoring model weights from the end of the best epoch: '
                        f'{self.best_epoch + 1}.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f'Epoch {self.stopped_epoch + 1}: early stopping')

    def get_monitor_value(self, logs):
        logs = logs or {}

        monitor_value = logs.get(self.monitor)
        monitor2_value = logs.get(self.monitor2)

        combined = (0.25*monitor_value) + (0.75*monitor2_value)

        if combined is None:
            logging.warning('Early stopping conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        return combined

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

