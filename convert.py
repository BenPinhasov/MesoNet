import tensorflow as tf
from classifiers import Meso4
import tf2onnx

keras_model = Meso4()
keras_model.load('weights/Meso4_DF.h5')
keras_model = keras_model.model


spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)
output_path = keras_model.name + ".onnx"

model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature=spec, output_path=output_path)




