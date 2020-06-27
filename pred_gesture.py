import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import pathlib
import random
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 形态学开操作
def open_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)  # 开操作
    return dst


# 形态学闭操作
def close_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)  # 开操作
    return dst


# 形态学腐蚀操作
def erode_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.erode(binary, kernel)  # 腐蚀
    return dst


# 形态学膨胀操作
def dilate_binary(binary, x, y):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))  # 获取图像结构化元素
    dst = cv.dilate(binary, kernel)  # 膨胀返回
    return dst


def nothing(x):
    pass


def creatTrackbar():

    cv.createTrackbar("x1", "roi_adjust", 200, 800, nothing)
    cv.createTrackbar("x2", "roi_adjust", 400, 800, nothing)
    cv.createTrackbar("y1", "roi_adjust", 100, 800, nothing)
    cv.createTrackbar("y2", "roi_adjust", 300, 800, nothing)


def get_roi(frame, x1, x2, y1, y2):
    dst = frame[y1:y2, x1:x2]
    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    return dst


def body_detetc(frame):

    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)  # Ycrcb 色彩空间 分割肤色
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])
    mask = cv.inRange(ycrcb, lowerb=lower_ycrcb, upperb=upper_ycrcb)  # ycrcb 掩码
    return mask




if __name__ == "__main__":

    capture = cv.VideoCapture(0)
    creatTrackbar()
    channels = 3
    DEFAULT_FUNCTION_KEY = "serving_default"

    # 加载已经训练好的数据集的权值
    loaded = tf.saved_model.load('./gesture_recognition_model/gestureModel_one/')
    network = loaded.signatures[DEFAULT_FUNCTION_KEY]
    
    print(list(loaded.signatures.keys()))
    print('加载 weights 成功')
   
    # 图片预测  
    # a=[]
    # for i in range(0,10):
    #     path =  "./test_gesture_data/" + str(i) + "/1517.jpg"
    #     print(path)
    #     image = tf.io.read_file(path)
    #     image = tf.image.decode_jpeg(image, channels=channels)
    #     image = tf.image.resize(image, [100, 100])
    #     image1 = image / 255.0  # normalize to [0,1] range
    #     image1 = tf.expand_dims(image1, axis=0)
    #     # print(image1.shape)
        
    #     pred = network(image1)
    #     # print("预测结果原始结果", pred)
    #     pred = tf.nn.softmax(pred['output_1'], axis=1)
    #     # print("预测softmax后", pred.numpy())
    #     pred = tf.argmax(pred, axis=1)
    #     # print("图片中手势识别结果为：", pred.numpy()[0])
    #     a.append(pred.numpy()[0])
    # for i in range(0, len(a)):
    #     print(a[i], end=" ")
    path =  "./test_gesture_data/8/1517.jpg"
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=channels)
    image = tf.image.resize(image, [100, 100])
    image1 = image / 255.0  # normalize to [0,1] range
    image1 = tf.expand_dims(image1, axis=0)
    # print(image1.shape)
    
    pred = network(image1)
    # print("预测结果原始结果", pred)
    pred = tf.nn.softmax(pred['output_1'], axis=1)
    # print("预测softmax后", pred.numpy())
    pred = tf.argmax(pred, axis=1)
    print("图片中手势识别结果为：", pred.numpy()[0])

    img = cv.imread(path)
    AddText = img.copy()
    text = str(pred.numpy()[0])
    cv.putText(AddText, text, (15, 25), cv.FONT_HERSHEY_COMPLEX, 1, (250, 0, 0), 4)
    
    cv.imshow('123',AddText)
    cv.waitKey(0)
    
