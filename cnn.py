from keras import layers
from keras import models
from keras import optimizers


class CNN(object):
    def __init__(self):
        print('init')
        self.model=None
        self.train_generator=None
        self.validation_generator=None


    def createNET(self):
        '''
        定义网络模型
        :return:
        '''
        model = models.Sequential()
        '''
        32: filter 个数
        (3,3) fitler 大小
        (150, 150, 3)：输入图片大小 150*150  3 彩色图片
        output shape （150-（3-1），150-（3-1），filter个数）
        '''
        model.add(layers.Conv2D(32, (3, 3),activation='relu',input_shape=(150, 150, 3)))

        '''
        output shape (输入\2,输入\2)
        '''
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        '''
        return 成1维的数据
        '''
        model.add(layers.Flatten())
        #增加drop层
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
        model.summary()
        self.model=model

    def train(self):
        '''
         训练
        '''
        '''
        epochs:训练次数
        steps_per_epoch：将一个epoch分为多少个steps，也就是划分一个batch_size多大 比如steps_per_epoch=10，则就是将训练集分为10份
        validation_steps：当steps_per_epoch被启用的时候才有用，验证集的batch_size
        '''
        history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=100,
            validation_data=self.validation_generator,
            validation_steps=50)
        # 保存模型
        self.model.save('cats_and_dogs_small_1.h5')
        return history

if __name__ == '__main__':
    cnn=CNN()
