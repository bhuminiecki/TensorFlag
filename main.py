import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from config import IMG_HEIGHT, IMG_WIDTH, TRAIN_DIR, TRAIN_SIZE, TEST_DIR, TEST_SIZE, EPOCHS, CLASS_NAMES
from sklearn.metrics import confusion_matrix
from time import gmtime, strftime

def generate_data(image):
    orb = cv.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    return des


# CREATE DATA GENERATORS

train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=45,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           fill_mode="constant"
                                           )
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=TRAIN_SIZE,
                                                           directory=TRAIN_DIR,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='sparse')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=TEST_SIZE,
                                                              directory=TEST_DIR,
                                                              shuffle=False,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='sparse')

# CREATE MODEL

model = keras.Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(28, activation="softmax")
])

# COMPILE MODEL

model.compile(optimizer=adam(learning_rate=0.001, amsgrad=True), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# TRAIN MODEL

history = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=EPOCHS)

# SAVE MODEL

model.save("output/model.h5")

# VISUALISE LEARNING PROCESS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

progress = plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

Y_pred = model.predict_generator(val_data_gen)
y_pred = np.argmax(Y_pred, axis=1)
confused = confusion_matrix(val_data_gen.classes, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(confused)

ax.set_xticks(np.arange(len(CLASS_NAMES)))
ax.set_yticks(np.arange(len(CLASS_NAMES)))

ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        text = ax.text(j, i, confused[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()

time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
progress.savefig("output/visualisation/progress-"+time+".jpg")
fig.savefig("output/visualisation/confusionmattrix-"+time+".jpg")
