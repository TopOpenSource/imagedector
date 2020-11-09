# keras 提供了一些预训练模型，也就是开箱即用的 已经训练好的模型
# 我们可以使用这些预训练模型来进行图像识别，目前的预训练模型大概可以识别2.2w种类型的东西
# 可用的模型：
# VGG16
# VGG19
# ResNet50
# InceptionResNetV2
# InceptionV3
# 这些模型被集成到 keras.applications 中
# 当我们使用了这些内置的预训练模型时，模型文件会被下载到 ~/.keras/models/并在载入模型时自动载入
# VGG16，VGG19，ResNet50 默认输入尺寸是224x224
# InceptionV3， InceptionResNetV2 模型的默认输入尺寸是299x299

# 使用内置的预训练模型的步骤
# step1 导入需要的模型
# step2 将需要识别的图像数据转换为矩阵（矩阵的大小需要根据模型的不同而定）
# step3 将图像矩阵丢到模型里面进行预测
# ----------------------------------------------------------------------------------
# step1
import cv2
import numpy as np
from getConfig import getOption
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input


class ImageTools(object):
    """
    使用keras预训练模型进行图像识别
    """

    def __init__(self, img, model, w):
        self.image = img
        self.model = model
        self.weight = w

    # step2
    def image2matrix(self, img):
        """
        将图像转为矩阵
        """
        image = cv2.imread(img)
        image = cv2.resize(image, self.dim)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    @property
    def dim(self):
        """
        图像矩阵的维度
        """
        if self.model in ["inceptionv3", "inceptionresnetv2"]:
            shape = (299, 299)
        else:
            shape = (224, 224)

        return shape

    @property
    def Model(self):
        """
        模型
        """
        models = {
            "vgg16": VGG16,
            "vgg19": VGG19,
            "resnet50": ResNet50,
            "inceptionv3": InceptionV3,
            "inceptionresnetv2": InceptionResNetV2
        }

        return models[self.model]

    # step3
    def prediction(self):
        """
        预测
        """
        model = self.Model(weights=self.weight)
        if self.model in ["inceptionv3", "inceptionresnetv2"]:
            preprocess = preprocess_input(self.image2matrix(self.image))
        else:
            preprocess = imagenet_utils.preprocess_input(self.image2matrix(self.image))

        predict = model.predict(preprocess)

        decode_predict = decode_predictions(predict)

        for (item, (imgId, imgLabel, proba)) in enumerate(decode_predict[0]):
            print("{}, {}, {:.2f}%".format(item + 1, imgLabel, proba * 100))


if __name__ == "__main__":
    image = getOption("image", "image_path")
    model = getOption("model", "model")
    weight = getOption("weights", "weight")
    tools = ImageTools(image, model, weight)
    tools.prediction()