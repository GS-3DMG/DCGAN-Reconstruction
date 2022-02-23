'''
对三维数据进行处理并且最终输出为一个numpy.array类型的三维矩阵(应该要三个维度的大小一样才可以卷积)
主要函数：

1.read_3Dfile(filepath)：进行文件的读取并返回一个三维的array
2.standardizate_arr(data_arr):进行三维数组的标准化，将三个轴的大小进行统一（以最小的标准进行统一）p.s思考后决定不适用直接补全为256因为会确实数据较多轴的数据的连续性
3.standardizate_to_256(data_arr,mode = 0):进行标准化后array的裁剪或是填充最终返回一个256*256*256的array
4.handle_3Ddata(trainning_data_path,label_data_path):进行函数的综合，进行3D文件的处理（txt格式）
5.read_npy_trainning_3Dimage(filepath):训练数据的读取（读取生成的trainning_3Dimage等）

直接将输入的数据变为单个文件的路径，批量的处理写在外面
'''

import numpy as np
import cv2
import pandas as pd
import matplotlib as plt
import math

'''
对txt文档进行读取并且返回一个array的三维数组(mat类型不能使用三维)
'''
def read_3Dfile(filepath):
    data_sum = []
    f = open(filepath)
    i = 0
    for line in f.readlines():
        dataset = line.strip().split()
        if len(dataset) == 0:
            continue
        data_sum.append(dataset)

        i = i+ 1
    data_arr = np.array(data_sum)
    #data_arr = data_arr[1:]
    data_arr = data_arr.reshape([64,64,64])
    #print(data_arr)
    return data_arr

'''
进行筛选确保进入网络的为大小规格相同的三维arr
'''
def standardizate_arr(data_arr):
    #data_arr = np.array(data_arr)
    x,y,z = data_arr.shape
    print(x)
    if x == y and x == z:
        data_arr_stan = data_arr
    else:
        min_num = min(x,y,z)
        #print('最小的维度数值为：',min_num)
        data_arr_stan = data_arr[0:min_num,0:min_num,0:min_num]
        #print("整形以后的arr：",data_arr_stan)
        x,y,z = data_arr_stan.shape
        #print("整形以后的arr的维度：",x,y,z)
    return data_arr_stan


'''
进行标准化array的扩充或裁剪(两个模式):将其标准化为相应的输入维度的大小
1.关键点较少并且关键点的矢量位置很重要mode = 0：直接由0来填充多余的点
2.二值连续图像mode = 1（默认）：使用resize函数进行扩充
'''
def standardizate_to_inputdim(data_arr,standard_x,standard_y,standard_z,mode = 0):
    x_num = data_arr.shape[0]
    y_num = data_arr.shape[1]
    z_num = data_arr.shape[2]
    if mode == 1:
        output_arr = np.resize(data_arr,(standard_x,standard_y,standard_z))
    else:
        if x_num < standard_x:
            missing_num = standard_x - x_num
            fill_arr_x = np.zeros([missing_num,y_num,z_num])#用于x轴的补充
            output_arr = np.concatenate([data_arr, fill_arr_x], axis=0)  # 补充x轴
        else:
            output_arr = data_arr[0:standard_x]
        print("进行第一维度处理后的output_arr = ",output_arr.shape)


        if y_num < standard_y:
            missing_num = standard_y - y_num
            fill_arr_y = np.zeros([standard_x,missing_num,z_num])#用于y轴的补充
            output_arr = np.concatenate([output_arr, fill_arr_y], axis=1)  # 补充y轴
        else:
            output_arr = output_arr[:, 0:standard_y]
        print("进行第二唯独处理过后的output_arr= ",output_arr.shape)


        if z_num < standard_z:
            missing_num = standard_z - z_num
            fill_arr_z = np.zeros([standard_x,standard_y,missing_num])#用于z轴的补充
            output_arr = np.concatenate([output_arr,fill_arr_z],axis=2)#补充z轴
        else:
            output_arr = output_arr[:,:, 0:standard_z]
            #不知道为啥最终输出的数值有0.0，可能有影响

    print("00" * 100)
    print("输入结果的维度 = ",output_arr.shape)
    print("00" * 100)
    return output_arr


'''
进行数据处理并返回相应的数据参数
'''

def handle_3Ddata(trainning_data_path,label_data_path,trainning_data_name,training_data_format,label_data_format,standard_x,standard_y,standard_z,RGB_data = False):#直接将输入的数据变为单个文件的路径，批量的处理写在外面

    trainning_data_path = trainning_data_path + trainning_data_name + training_data_format
    label_data_path = label_data_path + trainning_data_name + label_data_format
    print("训练图像路径 = ",trainning_data_path)
    print("训练标签图像路径 = ",label_data_path)
    trainning_data_arr = read_3Dfile(trainning_data_path)
    trainning_data_arr = standardizate_arr(trainning_data_arr)
    trainning_data_arr = standardizate_to_inputdim(trainning_data_arr,standard_x,standard_y,standard_z,mode=0)
    label_data_arr = read_3Dfile(label_data_path)
    label_data_arr = standardizate_arr(label_data_arr)
    label_data_arr = standardizate_to_inputdim(label_data_arr,standard_x,standard_y,standard_z,mode=0)
    #经过处理后输入数据的大小均为256*256*256便于网络进行统一训练

    #进行判断是否输入的三维图像为彩色图（三通道就不用填充维度），单通道的灰度图像需要进行最后一维的填充到3
    #进行维度填充
    if RGB_data == False:
        trainning_data_arr = np.expand_dims(np.array(trainning_data_arr).astype(np.float32),axis=3)
        trainning_data_arr = np.concatenate([trainning_data_arr, trainning_data_arr, trainning_data_arr], axis=3)
        label_data_arr = np.expand_dims(np.array(label_data_arr).astype(np.float32),axis=3)
        label_data_arr = np.concatenate([label_data_arr, label_data_arr, label_data_arr], axis=3)

    #进行归一化处理（便于收敛）
    trainning_data_arr = trainning_data_arr / 127.5 - 1.
    label_data_arr = label_data_arr / 127.5 - 1

    return trainning_data_arr,label_data_arr







# if __name__ == "__main__":
#     data_arr = read_3Dfile('D:/PycharmProjects/Generate_3D_GAN/test_3D_data.txt')
#     data_arr = standardizate_arr(data_arr)
#     data_arr = standardizate_to_256(data_arr,mode=0)
#     print("standardizate_to_256的data_arr：",data_arr,data_arr.shape,type(data_arr))