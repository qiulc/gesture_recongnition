import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import csv
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 获取图片，存放到对应的列表中，同时贴上标签，存放到label列表中
def get_files(file_dir):
    # 存放图片类别和标签的列表：第0类
    list_0 = []
    label_0 = []
    # 存放图片类别和标签的列表：第1类
    list_1 = []
    label_1 = []
    # 存放图片类别和标签的列表：第2类
    list_2 = []
    label_2 = []
    # 存放图片类别和标签的列表：第3类
    list_3 = []
    label_3 = []
    # 存放图片类别和标签的列表：第4类
    list_4 = []
    label_4 = []
    # 存放图片类别和标签的列表：第5类
    list_5 = []
    label_5 = []
    # 存放图片类别和标签的列表：第6类
    list_6 = []
    label_6 = []
    # 存放图片类别和标签的列表：第6类
    list_7 = []
    label_7 = []
    # 存放图片类别和标签的列表：第8类
    list_8 = []
    label_8 = []
    # 存放图片类别和标签的列表：第9类
    list_9 = []
    label_9 = []

    for file in os.listdir(file_dir):  # 获得file_dir路径下的全部文件名
        # 拼接出图片文件路径
        image_file_path = os.path.join(file_dir, file)
        for image_name in os.listdir(image_file_path):
            # print('image_name',image_name)
            # 图片的完整路径
            image_name_path = os.path.join(image_file_path, image_name)
            # print('image_name_path',image_name_path)
            # 将图片存放入对应的列表
            if image_file_path[-1:] == '0':
                list_0.append(image_name_path)
                label_0.append(0)
            elif image_file_path[-1:] == '1':
                list_1.append(image_name_path)
                label_1.append(1)
            elif image_file_path[-1:] == '2':
                list_2.append(image_name_path)
                label_2.append(2)
            elif image_file_path[-1:] == '3':
                list_3.append(image_name_path)
                label_3.append(3)
            elif image_file_path[-1:] == '4':
                list_3.append(image_name_path)
                label_3.append(4)
            elif image_file_path[-1:] == '5':
                list_3.append(image_name_path)
                label_3.append(5)
            elif image_file_path[-1:] == '6':
                list_3.append(image_name_path)
                label_3.append(6)
            elif image_file_path[-1:] == '7':
                list_3.append(image_name_path)
                label_3.append(7)
            elif image_file_path[-1:] == '8':
                list_3.append(image_name_path)
                label_3.append(8)
            else:
                list_4.append(image_name_path)
                label_4.append(9)

    
    # 合并数据
    image_list = np.hstack((list_0, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8, list_9))
    label_list = np.hstack((label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9))
    # 利用shuffle打乱数据
    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置
    np.random.shuffle(temp)

    # 将所有的image和label转换成list
    image_list = list(temp[:, 0])
    image_list = [i for i in image_list]
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    # print(image_list)
    # print(label_list)
    return image_list, label_list


def get_tensor(image_list, label_list):
    ims = []
    for image in image_list:
        # 读取路径下的图片
        x = tf.io.read_file(image)
        # 将路径映射为照片,3通道
        x = tf.image.decode_jpeg(x, channels=3)
        # 修改图像大小
        x = tf.image.resize(x, [32, 32])
        # 将图像压入列表中
        ims.append(x)
    # 将列表转换成tensor类型
    img = tf.convert_to_tensor(ims)
    y = tf.convert_to_tensor(label_list)
    return img, y


def train_model(train_data):
    # 构建模型
    network = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)])
    network.build(input_shape=(None, 100, 100, channels))
    network.summary()
    
    network.compile(optimizer=optimizers.SGD(lr=0.001),
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy']
                    )
    checkpoint_filepath = "./gesture_recognition_model/gestureModel"
    callbacks = [
        # 保存模型的回调函数
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,  # 保存路径
                                           save_weights_only=True,
                                           verbose=0,
                                           save_freq='epoch'),  # 保存频次以周期频次来计算
        # 中止训练的回调函数
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)  # 防止过拟合,如果过拟合了之后就终止1_val验证集上loss变高终止观察3个周期
    ]
    # 只要在model.fit训练模型里面加上 callbacks=callbacks  这个参数,那在训练模型的时候就会按  照我们设计的回调函数来保存模型
    # 模型训练
    network.fit(train_data, epochs=5, callbacks=callbacks)  # 因为是dataset数据集是个元组自带标签所以不用分x和y了
    # network.evaluate(test_data)
    network.save_weights('./gesture_recognition_model/gestureModel_one.h5')
    print("保存模型权重成功")
    tf.saved_model.save(network, './gesture_recognition_model/gestureModel_one')
    print("保存模型成功")
    return network
 

def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [100, 100])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    return preprocess_image(image), label

if __name__ == "__main__":

    # 训练图片的路径
    global channels
    channels = 3
    train_dir = './train_gesture_data/'
    test_dir = './test_gesture_data/'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # 训练图片与标签
    image_list, label_list = get_files(train_dir)
    # 测试图片与标签
    test_image_list, test_label_list = get_files(test_dir)

    # for i in range(len(image_list)):
    # print('图片路径 [{}] : 类型 [{}]'.format(image_list[i], label_list[i]))
    x_train, y_train = get_tensor(image_list, label_list)
    x_test, y_test = get_tensor(test_image_list, test_label_list)

    
    # 载入训练数据集
    db_train = tf.data.Dataset.from_tensor_slices((image_list, y_train))
    # # shuffle:打乱数据,map:数据预处理，batch:一次取喂入10样本训练
    db_train = db_train.shuffle(1000).map(load_and_preprocess_image).batch(10)

    print("dataset", db_train)
    network = train_model(db_train)

