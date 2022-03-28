import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt


# TODO: Нужно заставить дома работать на GPU.
#  Пока что ругается и дулает на основном процессоре


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


def transform_gray(image):  # Перевод иозбражения из цветной в ч/б
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
    return gray_image


# ---> Нобор и создание датасета <---
PATH = os.getcwd()
print('[+] PATH:', PATH)
x = []
y = []

os.chdir(PATH + '/dataset/PetImages')
count = 0
for dir in os.listdir():
    os.chdir(dir)
    for file in os.listdir()[:500]:
        # print('>>>', file)
        if '.db' == file:
            continue
        image = cv2.imread(file)
        if image is None:
            continue
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_2 = scalePicture(image, 299, 299)
        x.append(np.array(image_2).astype('float32'))
        y.append(count)

    os.chdir('..')
    count += 1

x = np.asarray(x)
y = np.asarray(y)
x = tf.keras.applications.inception_v3.preprocess_input(x, data_format=None)

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

# ---> Обучение модели <---
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),  # 'categorical_crossentropy'
              metrics=[keras.metrics.BinaryAccuracy()])

history = model.fit(X_train,
                    y_train,
                    epochs=1,
                    validation_data=(X_test, y_test))

# TODO: Посмотреть, как правильно сохранять модель в файл.
#  Пока ругается на какую то чепушню...
# history.save_model('model.h5')

model.summary()
print(history.history.keys())
# summarize history for accuracy (jupyter only)
# TODO: Посмотреть, как лучше строить окно и делать графики.
#  Что-то не вспомнилось сразу...

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(history.history['binary_accuracy'])
ax.plot(history.history['val_binary_accuracy'])  # RAISE ERROR
# ax.title('model accuracy')
ax.set_ylabel('accuracy')
ax.set_xlabel('epoch')
ax.legend(['train', 'test'], loc='upper left')
plt.show()
plt.close()

# fig = plt.figure()
ay = fig.add_subplot(111)
ay = plt.plot(history.history['loss'])
ay.plot(history.history['val_loss'])  # RAISE ERROR
# ay.title('model loss')
ay.set_ylabel('loss')
ay.set_xlabel('epoch')
ay.legend(['train', 'test'], loc='upper left')
plt.show()
plt.close()
