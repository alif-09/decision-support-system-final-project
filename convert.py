import tensorflow as tf
import pickle
import json

# Konversi model Keras ke TFLite
def convert_model_to_tflite(model_path, tflite_path):
    print("Converting model to TFLite...")
    model = tf.keras.models.load_model(model_path)
    
    # Initialize the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable Select TensorFlow Ops and experimental flags for complex models
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # TensorFlow Lite built-in ops
        tf.lite.OpsSet.SELECT_TF_OPS      # TensorFlow Select ops
    ]
    converter.experimental_enable_resource_variables = True
    converter._experimental_lower_tensor_list_ops = False

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model successfully converted to {tflite_path}!")

# Konversi tokenizer ke JSON
def convert_tokenizer_to_json(tokenizer_path, json_path):
    print("Converting tokenizer to JSON...")
    # Load tokenizer from pickle file
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Convert tokenizer to JSON format
    tokenizer_json = tokenizer.to_json()

    # Save JSON to file
    with open(json_path, 'w') as json_file:
        json.dump(tokenizer_json, json_file)

    print(f"Tokenizer successfully converted to {json_path}!")

# File paths
model_path = 'model.keras'
tflite_path = 'model.tflite'
tokenizer_path = 'tokenizer.pkl'
json_path = 'tokenizer.json'

# Run the converters
convert_model_to_tflite(model_path, tflite_path)
convert_tokenizer_to_json(tokenizer_path, json_path)
