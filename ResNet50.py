from keras import models, layers, optimizers
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions


import cv2
import numpy as np
import matplotlib.pyplot as plt

class ResNet50Demo(object):
    def createModel(self):
        conv_base = ResNet50(weights="imagenet", include_top=False, input_shape=(500, 400, 3), classes=2)
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        conv_base.trainable = False
        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
        return model

    def dataGenerator(self):
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True, )

        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\train',
            # All images will be resized to 150x150
            target_size=(500, 400),
            batch_size=5,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\validation',
            target_size=(500, 400),
            batch_size=5,
            class_mode='binary')

        return train_generator, validation_generator

    def test(self):
        model = models.load_model('ResNet50_weight.h5')
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow_from_directory(
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\test',
            target_size=(500, 400),
            batch_size=5,
            class_mode='binary')

        test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // 5)
        print('test acc:', test_acc)
        print('test_loss:', test_loss)


    def train(self):
       train_generator, validation_generator=self.dataGenerator()
       model=self.createModel()
       history = model.fit_generator(
           train_generator,
           steps_per_epoch=train_generator.samples // 5,
           epochs=19,
           validation_data=validation_generator,
           validation_steps=validation_generator.samples // 5)
       model.save('ResNet50_weight.h5')

       acc = history.history['acc']
       val_acc = history.history['val_acc']
       loss = history.history['loss']
       val_loss = history.history['val_loss']

       epochs = range(1, len(acc) + 1)

       plt.plot(epochs, acc, 'bo', label='Training acc')
       plt.plot(epochs, val_acc, 'b', label='Validation acc')
       plt.title('Training and validation accuracy')
       plt.legend()

       plt.figure()

       plt.plot(epochs, loss, 'bo', label='Training loss')
       plt.plot(epochs, val_loss, 'b', label='Validation loss')
       plt.title('Training and validation loss')
       plt.legend()

       plt.show()


    def predict(self):
        model=models.load_model("ResNet50_weight.h5")

        image = cv2.imread("D:/cat.jpg")
        image = cv2.resize(image, (400,500))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        preprocess = imagenet_utils.preprocess_input(image)
        predict = model.predict_classes(preprocess)
        #decode_predict = decode_predictions(predict)
        print(predict)



if __name__ == '__main__':
    resnet50=ResNet50Demo()
    resnet50.predict()
