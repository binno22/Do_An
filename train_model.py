# IMPORT THƯ VIỆN
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# KHỞI TẠO
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 32

DIRECTORY = '/content/drive/MyDrive/Final_Do_An/dataset'
CATEGORIES = ["with_mask", "without_mask"]

print('LOADING...')

data = []
labels = []

for category in CATEGORIES:
  path = os.path.join(DIRECTORY, category)
  for img in os.listdir(path):
    img_path = os.path.join(path, img)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(category)

print("GÁN NHÃN XONG")

# ONE-HOT
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# CHIA TẬP TRAIN VÀ TEST
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
print(X_train.shape, y_train.shape)

# TĂNG CƯỜNG HÌNH ẢNH
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# XÂY DỰNG MODEL
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# COMPILE MODEL
print("Compiling Model...")
opt = Adam(lr=LEARNING_RATE)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# TRAIN MODEL
print("Training Model...")
H = model.fit(
    aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BATCH_SIZE,
    epochs=EPOCHS
)

# DỰ ĐOÁN
predIdxs = model.predict(X_test, batch_size=BATCH_SIZE)
#  Tìm index của nhãn có xác suất dự đoán lớn nhất tương ứng
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(y_test.argmax(axis=1), predIdxs, target_names=lb.classes_))


# SAVE MODEL
model.save("mask_detector.model", save_format="h5")

# VẼ ĐỒ THỊ
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="val_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
