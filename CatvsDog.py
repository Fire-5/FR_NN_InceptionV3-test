import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib as plt
from keras.datasets import cifar10


def scale(img):                                                         #Перевод изображение в размер 40x40
    dim = (40, 40)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


def openPicture(name):
    image = cv2.imread(name)
    return image


def showPicture(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform_gray(image):                                              #Перевод иозбражения из цветной в ч/б
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
    return gray_image


print(os.getcwd())
PATH = os.getcwd()
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
        dim = (150, 150)
        image_2 = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        x.append(np.array(image_2).astype('float32'))
        y.append(count)

    os.chdir('..')
    count += 1

# print(x)

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    random_state=42)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(train.len)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, sep='\n')


base_model = keras.applications.Xception(
    weights='imagenet',  # загрузка предобученной модели
    input_shape=(150, 150, 3),
    include_top=False)

base_model.trainable = False

# Создайте новую модель сверху.
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

# Обучите модель новым данным.
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# history = model.fit(train_norm,
#                     trainY,
#                     epochs=5,
#                     batch_size=512,
#                     validation_data=(test_norm,testY),
#                     validation_batch_size=100)
history = model.fit(X_train,
                    y_train,
                    epochs=5,
                    validation_data=(X_test, y_test))

model.summary()


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  # RAISE ERROR
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #RAISE ERROR
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
