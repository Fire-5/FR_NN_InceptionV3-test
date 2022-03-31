import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt


# TODO: Нужно заставить дома работать на GPU.
#  Пока что ругается и делает на основном процессоре


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


def getDataset(path, lenght):
    # ---> Нобор и создание датасета <---
    dataset_value = []
    dataset_key = []
    os.chdir(path)
    count = 0

    for dir in os.listdir():
        os.chdir(dir)
        for file in os.listdir()[:lenght]:
            # print('>>>', file)
            if '.db' == file:
                continue
            image = cv2.imread(file)
            if image is None:
                continue
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_2 = scalePicture(image, 299, 299)
            dataset_value.append(np.array(image_2).astype('float32'))
            dataset_key.append(count)

        os.chdir('..')
        count += 1

    dataset_value = np.asarray(dataset_value)
    dataset_value = tf.keras.applications.inception_v3.preprocess_input(dataset_value, data_format=None)
    dataset_key = np.asarray(dataset_key)

    return dataset_value, dataset_key


PATH = os.getcwd()

x, y = getDataset(PATH + '/dataset/PetImages/', lenght=1000)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('Shape:', X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n[+] > ')

# ---> Создание модели <---
input_tensor = keras.layers.Input(shape=(299, 299, 3))
base_model = keras.applications.InceptionV3(
    include_top=False,  # включать ли полносвязный слой вверху как последний слой сети
    weights="imagenet",  # предварительное обучение или путь к загружаемому файлу весов
    input_tensor=input_tensor,  # необязательный тензор для использования в качестве входного изображения для модели
    input_shape=None,  # необязательный кортеж формы входного изображения
    pooling=None,  # дополнительный режим объединения для извлечения признаков (avg,max)
    classes=2,  # необязательное количество классов для классификации изображений
    classifier_activation="softmax",  # Функция активации для использования на «верхнем» слое
)

# ---> Создание новой модели сверху. <---
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
predictions = keras.layers.Dense(1, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input,
                           outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


# model = base_model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')

# ---> Обучение модели <---
history = model.fit(X_train,
                    y_train,
                    epochs=3,
                    batch_size=50,
                    validation_data=(X_test, y_test))

# ---> Тонкая настройка модели <---
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

history = model.fit(X_train,
                    y_train,
                    epochs=3,
                    batch_size=50,
                    validation_data=(X_test, y_test))

model.save(PATH + '/my_model.h5')

# history = model.load(PATH + '/my_model.h5')

model.summary()
print(history.history.keys())
# summarize history for accuracy (jupyter only)

plt.style.use('_mpl-gallery')

fig, ax = plt.subplots(2, 1)

# ax[0].plot(history.history['binary_accuracy'], history.history['val_binary_accuracy'])  # RAISE ERROR
ax[0].plot(history.history['loss'], history.history['val_loss'])  # RAISE ERROR

ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[1].set_ylabel('loss')
ax[0].legend(['train', 'test'], loc='upper left')

fig.tight_layout()
plt.show()
plt.close()
