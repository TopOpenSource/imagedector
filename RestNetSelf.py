from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, AveragePooling2D, ZeroPadding2D,MaxPooling2D
from keras.layers import add, Flatten, Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt

class restNet(object):
    def Conv2d_BN(self,x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
        #数据标准化
        x = BatchNormalization(axis=3, name=bn_name)(x)
        #激活函数
        x = Activation('relu')(x)
        return x

    def Conv_Block(self,inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = self.Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x

    def creatcnn(self):
        inpt = Input(shape=(224, 224, 3))
        #在图片边缘添加 3*3的空矩阵
        x = ZeroPadding2D((3, 3))(inpt)

        x = self.Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')

        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = self.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

        x = self.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = self.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        x = self.Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        return model

    def createResultImage(self,history):
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

    def train(self):
        model = self.creatcnn()
        model.summary()
        # 保存模型结构图
        # plot_model(model, to_file='ResNet50_model.png', show_shapes=True)
        sgd = SGD(decay=0.0001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\train',
            target_size=(224, 224),
            batch_size=10,
            class_mode='categorical')

        test_datagen = ImageDataGenerator()
        validation_generator = test_datagen.flow_from_directory(
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\validation',
            target_size=(224, 224),
            batch_size=10,
            class_mode='categorical')

        history=model.fit_generator(train_generator,
                            steps_per_epoch=300,
                            epochs=100,
                            validation_data=validation_generator,
                            validation_steps=24,
                            verbose=1)

        self.createResultImage(history)

        test_generator = test_datagen.flow_from_directory(
            'D:\\data\\dogs-vs-cats\\cats_and_dogs_small\\test',
            target_size=(224, 224),
            batch_size=10,
            class_mode='categorical')

        test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_generator.samples // 10)
        print('test acc:', test_acc)
        print('test loss:', test_loss)


        model.save_weights('ResNet_weight.h5')


if __name__ == "__main__":
    restNet = restNet()
    restNet.train()




