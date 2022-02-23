'''
进行对原始数据的处理并最终生成相应的可训练的三维array的数据（三维的128*128*128或者64*64*64）
也就是对原始的较大的数据进行处理变为多个可以被训练的三维数据
def read_profile(filepath):进行数据的读取
prepar_profile_data(data_arr,label_arr):进行数据的填充处理
'''

from numpy import *
import pandas as pd
import tensorflow as tf
import random as random
import argparse
import os as os

parser = argparse.ArgumentParser(description='')

parser.add_argument("--full_pixel_value", default=0, help="value to full no profile")  #填充没有剖面的地方的像素所使用的值
parser.add_argument("--out_dim",default=64,help="output pixel dim")#想要输出的最终数据大小（128或者是64）
parser.add_argument("--data_dim_x",default=120,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--data_dim_y",default=150,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--data_dim_z",default=180,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--out_dir",default="D:/PycharmProjects/Generate_3D_GAN/dataset/train_label/",help='dir of out image')#向相应文件夹中输入数据(一般为test_3Dimage和train_3Dimage# )D:/PycharmProjects/Generate_3D_GAN/dataset/train_label/
#D:/PycharmProjects/Generate_3D_GAN/dataset/train_3Dimage/
parser.add_argument("--input_dir",default="D:/PycharmProjects/Generate_3D_GAN/fold_categorical_180x150x120.txt",help='dirs of input image3Ddata')#进行输入的原始数据文件#D:/PycharmProjects/Generate_3D_GAN/fold_categorical_180x150x120.txt
#D:/PycharmProjects/Generate_3D_GAN/dataset/profile_data/3.txt



args = parser.parse_args()  # 用来解析命令行参数


'''
进行对原始数据的读取
此数据的格式为全部为对应的属性像素值，直接按行列逐行读入就行
'''
def read_Primitive_data(filepath):
    data_sum = []
    f = open(filepath)

    for line in f.readlines():
        dataset = line.strip().split()
        data_sum.append(dataset)

        if len(dataset) == 1:
            continue
    data_arr = array(data_sum)
    data_arr = data_arr.reshape([args.data_dim_x,args.data_dim_y,args.data_dim_z])


    return data_arr #输出的为位置的信息以及每个位置相应的label


'''
进行类似滤波的处理，逐层进行移动并最终输出构造完毕的三维图像
输出一个128*128*128的三维图像
'''
def prepar_data(data_arr,name_No):
    out_prepar_data = zeros([args.out_dim*args.out_dim*args.out_dim])
    min_pixel_num = min(args.data_dim_x,args.data_dim_y,args.data_dim_z)
    cycle_num = min_pixel_num - args.out_dim
    print("待循环的次数为：",cycle_num)
    #name_No = 0
    for i in range(0,cycle_num):
        for j in range(0,cycle_num):
            for k in range(0,cycle_num):
                # 进行数组的提取
                print("第i:j:k循环 = ",i,j,k)
                out_prepar_data = data_arr[i:i+args.out_dim, j:j+args.out_dim,k:k+args.out_dim]
                #out_prepar_data = data_arr[i:, j:, k:]
                print(out_prepar_data,out_prepar_data.shape)
                name_No = name_No + 1

                # 进行保存(两个方法：一个为直接存储npy文件，另一个为存储为普通的txt文件)
                #第一个
                # savepath = args.out_dir + str(name_No) + ".npy"
                # save(out_prepar_data,savepath)
                #第二个
                savepath = args.out_dir + str(name_No) + ".txt"
                print("最终的存储路径 = ",savepath)
                # f = open(savepath, 'a')
                # f.write(str(128) + '\t' + str(128) + '\t' + str(128))
                # f.close()
                f = open(savepath, 'a')
                print("各种维度的总数量大小",out_prepar_data.shape[0],out_prepar_data.shape[1],out_prepar_data.shape[2])
                for m in range(0,out_prepar_data.shape[0]):
                    for n in range(0,out_prepar_data.shape[1]):
                        for s in range(0,out_prepar_data.shape[2]):

                            if out_prepar_data[m][n][s] == str(1):
                                f.write('\n' + str(255))
                            if out_prepar_data[m][n][s] == str(0):
                                f.write('\n' + str(out_prepar_data[m][n][s]))
                f.close()



if __name__ == "__main__":
    data_arr = read_Primitive_data(args.input_dir)
    print("原始数据的大小及属性 = ",data_arr,data_arr.shape)
    prepar_data(data_arr,0)
    # print(data_arr[0][1])
    # complete_profile_data = prepar_profile_data(data_arr,data_label)
    # print('填充给完毕后的含剖面训练准备数据：', complete_profile_data, complete_profile_data.shape)