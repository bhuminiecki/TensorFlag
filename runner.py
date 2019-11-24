import tensorflow as ts
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH, INPUT_DIR, INPUT_SIZE, CLASS_NAMES

input_image_generator = ImageDataGenerator(rescale=1. / 255)

input_data_gen = input_image_generator.flow_from_directory(batch_size=INPUT_SIZE,
                                                           directory=INPUT_DIR,
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH))

model = keras.models.load_model("output/model.h5")

results = []

prediction = model.predict(input_data_gen)
for entry in prediction:
    print("======================================")
    lst = list(sorted(range(len(entry)), key = lambda sub: entry[sub])[-3:])
    lst.reverse()
    for guess in lst:
        print(CLASS_NAMES[guess], entry[guess])

