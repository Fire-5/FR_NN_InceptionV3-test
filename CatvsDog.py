import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def scalePicture(img, x, y):
    dim = (x, y)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def openPicture(name):
    image = cv2.imread(name)
    return image


def showPicture(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
    return gray_image

# ---> Нобор и создание датасета <---
# Полезное чтиво
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
PATH = os.getcwd()

train_data_dir = PATH + '/dataset/train'
validation_data_dir = PATH + '/dataset/validation'
img_width, img_height = 150, 150
batch_size = 16

train_datagen = ImageDataGenerator(
      rescale=1./255, # изменения по размерам изображения
      rotation_range=40,  # диапазон, в пределах которого можно произвольно поворачивать изображения
      width_shift_range=0.2,  # это диапазоны (в виде доли от общей ширины или высоты), в пределах которых можно
      height_shift_range=0.2, # произвольно перемещать изображения по вертикали или горизонтали
      shear_range=0.2,  # предназначен для случайного применения преобразований сдвига
      zoom_range=0.2,  # предназначен для случайного масштабирования внутри изображений
      horizontal_flip=True, # предназначен для случайного переворачивания половины изображений по горизонтали
      fill_mode='nearest' # заполнения вновь созданных пикселей, которые могут появиться после поворота или сдвига
)
test_datagen = ImageDataGenerator(rescale=1./255,)

train_gen = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# ---> Создание модели <---
input_tensor = keras.layers.Input(shape=(150, 150, 3))
base_model = keras.applications.InceptionV3(
    include_top=False,  # включать ли полносвязный слой вверху как последний слой сети
    weights="imagenet",  # предварительное обучение или путь к загружаемому файлу весов
    input_tensor=input_tensor,  # необязательный тензор для использования в качестве входного изображения для модели
    input_shape=(150, 150, 3),  # необязательный кортеж формы входного изображения
    pooling=None,  # дополнительный режим объединения для извлечения признаков (avg,max)
    classes=2,  # необязательное количество классов для классификации изображений
    classifier_activation="softmax",  # Функция активации для использования на «верхнем» слое
)
# ---> Делаем все слои необучаемыми <---
for layer in base_model.layers:
    layer.trainable = False

# ---> Создание новой модели сверху. <---
# x = base_model.output
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dense(1024, activation='relu')(x)
# predictions = keras.layers.Dense(1, activation='softmax')(x)

x = keras.layers.Flatten()(base_model.output)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.models.Model(inputs=base_model.input,
                           outputs=predictions)

model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ---> Обучение модели <---
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=test_gen
)

# history.save(PATH + '/my_model.h5')
# history.save_weights(PATH + '/first_try.h5')

# history = model.load(PATH + '/my_model.h5')

# model.summary()
print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.legend()

# plt.style.use('_mpl-gallery')
# fig, ax = plt.subplots(2, 1)
#
# ax[0].plot(history.history['acc'], history.history['val_acc'])  # RAISE ERROR
# ax[1].plot(history.history['loss'], history.history['val_loss'])  # RAISE ERROR
#
# ax[0].set_ylabel('acc')
# ax[0].set_xlabel('epoch')
# ax[1].set_ylabel('loss')
# ax[0].legend(['train', 'test'], loc='upper left')
#
# fig.tight_layout()

plt.show()
plt.close()
