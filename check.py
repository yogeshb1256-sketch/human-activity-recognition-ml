from tensorflow.keras.models import load_model

model = load_model("har_model_5class.h5")
print(model.input_shape)
