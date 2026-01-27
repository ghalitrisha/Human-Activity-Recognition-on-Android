import keras
import tensorflow as tf
import os
import tensorflow_model_optimization as tfmot
# from tensorflow_model_optimization.quantization.keras.quantizers import MovingAverageQuantizer, LastValueQuantizer


base_path = os.path.dirname(os.getcwd())


def representative_dataset(windows, take=1000, batch=1):
  for data in tf.data.Dataset.from_tensor_slices((windows)).batch(batch).take(take):
    yield [tf.dtypes.cast(data, tf.float32)]

def get_baseline_model(input_shape, num_classes):
    l2_reg = keras.regularizers.l2(1e-4)
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=5, padding="same", kernel_regularizer=l2_reg)(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", kernel_regularizer=l2_reg)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    
    dropout1 = keras.layers.Dropout(0.1)(conv2)



    lstm1 = keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=l2_reg)(dropout1)
    lstm2 = keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=l2_reg)(lstm1)


    dense1 = keras.layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(lstm2)
    dense1 = keras.layers.BatchNormalization()(dense1)

    dense1 = keras.layers.Dropout(0.35)(dense1)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense1)
    


    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_GRU_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)


    gru1 = keras.layers.GRU(64, return_sequences=True)(input_layer)
    # gru2 = keras.layers.GRU(64, return_sequences=True)(gru1)
    gru3 = keras.layers.GRU(32, return_sequences=False)(gru1)


    dense1 = keras.layers.Dense(128, activation="relu")(gru3)
    dense1 = keras.layers.BatchNormalization()(dense1)

    dense1 = keras.layers.Dropout(0.2)(dense1)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense1)


    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def get_dynamic_quantized_model(model, save_path = None):
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic_quant_model = converter.convert()
    if save_path:
        with open(save_path, "wb") as f_quant:
            f_quant.write(tflite_dynamic_quant_model)
    return tflite_dynamic_quant_model

def get_float_model(model, save_path=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    tflite_model = converter.convert()
    if save_path:
        with open(save_path, "wb") as f:
            f.write(tflite_model)
    return tflite_model
        
def get_int_16x8_quantized_model(model, representative_windows, save_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(representative_windows)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_int_quant_model = converter.convert()
    if save_path:
         with open(save_path, "wb") as f:
            f.write(tflite_int_quant_model)
    return tflite_int_quant_model



# class MyConvDenseConfig(tfmot.quantization.keras.QuantizeConfig):
#     def get_weights_and_quantizers(self, layer):
#         # Quantize kernel if present (Conv/Dense)
#         return [(layer.kernel, LastValueQuantizer(
#             num_bits=8, per_axis=True, symmetric=True, narrow_range=True
#         ))]

#     def get_activations_and_quantizers(self, layer):
#         return [(layer.activation, MovingAverageQuantizer(
#             num_bits=8, per_axis=False, symmetric=False, narrow_range=False
#         ))]

#     def set_quantize_weights(self, layer, quantize_weights):
#         layer.kernel = quantize_weights[0]

#     def set_quantize_activations(self, layer, quantize_activations):
#         layer.activation = quantize_activations[0]

#     def get_output_quantizers(self, layer):
#         []

#     def get_config(self):
#         return {}


# annotate ONLY Dense layers for QAT
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_apply

def get_qat_model_dense_only(input_shape, num_classes):
    inp = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv1D(64, 5, padding="same")(inp)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(128, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x); x = keras.layers.ReLU()(x)

    x = keras.layers.LSTM(128, return_sequences=True)(x)
    x = keras.layers.LSTM(64, return_sequences=False)(x)

    # QAT starts here (Dense only)
    x = quantize_annotate_layer(keras.layers.Dense(128, activation="relu"))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.1)(x)
    out = quantize_annotate_layer(keras.layers.Dense(num_classes, activation="softmax"))(x)

    annotated = keras.Model(inp, out)
    with tfmot.quantization.keras.quantize_scope():
        qat_model = quantize_apply(annotated)
    return qat_model


    

    

def get_pruned_model(model, begin_step, end_step, initial_sparsity=0.5, final_sparsity=0.8):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=initial_sparsity,final_sparsity=final_sparsity,begin_step=begin_step,end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    return model_for_pruning

class AnomalyDetector(keras.Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
      tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
      tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
    ])
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
      tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
      tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(6, activation="sigmoid"),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def get_autoencoder():
    autoencoder = AnomalyDetector()
    return autoencoder
