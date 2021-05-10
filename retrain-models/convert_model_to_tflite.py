import tensorflow as tf

saved_models_dir = "../models"
model_to_convert = "new_yamnet"
tflite_models_dir = "../models-tflite"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_models_dir + "/" + model_to_convert)  # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

tflite_model = converter.convert()

# Save the model.
with open(tflite_models_dir + "/" + model_to_convert + ".tflite", 'wb') as f:
    f.write(tflite_model)
