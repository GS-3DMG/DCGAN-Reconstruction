'''
进行对剖面数据的处理并最终生成相应的可训练的三维array的数据
训练所应用的数据就是利用此文件中的整理填充好的剖面数据进行训练
def read_profile(filepath):进行数据的读取
prepar_profile_data(data_arr,label_arr):进行数据的填充处理
'''

from numpy import *
import pandas as pd
import tensorflow as tf
import random as random
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument("--full_pixel_value", default=125, help="value to full no profile")  #填充没有剖面的地方的像素所使用的值
parser.add_argument("--data_dim_x",default=120,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--data_dim_y",default=150,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--data_dim_z",default=180,help="dim of expend 3D data x exis") #期望生成的不全后数据的x维度值
parser.add_argument("--out_dir",default="D:/PycharmProjects/Generate_3D_GAN/dataset/profile_data/",help='dir of out image')#向相应文件夹中输入数据(一般为test_3Dimage和train_3Dimage# )
parser.add_argument("--x_profile_file_path",default="D:/PycharmProjects/Generate_3D_GAN/profile_file/x_4.txt",help="filepath for x profile file")#对应的x方向的剖面数据的文件路径
parser.add_argument("--y_profile_file_path",default="D:/PycharmProjects/Generate_3D_GAN/profile_file/y_4.txt",help="filepath for x profile file")#对应的y方向的剖面数据的文件路径
parser.add_argument("--z_profile_file_path",default="D:/PycharmProjects/Generate_3D_GAN/profile_file/z_4.txt",help="filepath for x profile file")#对应的z方向的剖面数据的文件路径



args = parser.parse_args()  # 用来解析命令行参数


'''
进行对剖面数据的读取
此数据的格式为前三列记载的为数据坐标，而最后一列为相应的相1/0
'''
def read_profile(filepath):
    data_sum = []
    data_label = []
    f = open(filepath)
    i = 0
    for line in f.readlines():
        dataset = line.strip().split()
        data_sum.append([dataset[2],dataset[1],dataset[0]])
        data_label.append(int(float(dataset[-1])))

        if len(dataset) == 1:
            continue
        i = i + 1
    loc_data = array(data_sum)
    value_data = array(data_label)

    return loc_data,value_data #输出的为位置的信息以及每个位置相应的label

'''
构造带有剖面的三位图像数据（把对应的没有剖面的点进行补全）
将像素点进行填充
'''
def full_profile_pixel(loc_data_arr):
    num_sum = loc_data_arr.shape[0]
    print(num_sum)
    num_x = args.data_dim_x
    num_y = args.data_dim_y
    num_z = args.data_dim_z
    print("准备填充的三位数据大小",num_x,num_y,num_z)
    complete_profile_data = full([num_x,num_y,num_z],args.full_pixel_value)#进行填充，并且需要填充的值需要自己定义
    complete_profile_data = array(complete_profile_data)

    return complete_profile_data,num_sum



'''
将剖面数据整合到原有的一填充好的数据内
输出一个带有剖面的三维图像
'''
def prepar_profile_data(loc_data_arr,loc_label_arr,complete_profile_data,num_sum):

    for i in range(0,num_sum):
        x_i = loc_data_arr[i][0]
        y_i = loc_data_arr[i][1]
        z_i = loc_data_arr[i][2]
        print("包含的数据 ： ",x_i,y_i,z_i)
        print("对应的数据点",loc_label_arr[i])
        if loc_label_arr[i] == 1:
            loc_label_arr[i] = 255
        complete_profile_data[int(x_i)-1,int(y_i)-1,int(z_i)-1] = loc_label_arr[i]

    return complete_profile_data




'''
输入/构造钻井数据
'''
def make_well_data(complete_profile_data):
    pass


'''
输出填充好的数据到txt文档里面
'''
def save_txt_profile_3Dimage(complete_profile_data,name_No):
    savepath = args.out_dir + str(name_No) + ".txt"
    # print("最终的存储路径 = ", savepath)
    # f = open(savepath, 'a')
    # f.write(str(128) + '\t' + str(128) + '\t' + str(128))
    # f.close()
    print("各种维度的总数量大小", complete_profile_data.shape[0], complete_profile_data.shape[1], complete_profile_data.shape[2])
    for i in range(0,complete_profile_data.shape[0]):
        for j in range(0,complete_profile_data.shape[1]):
            for k in range(0,complete_profile_data.shape[2]):
                f = open(savepath, 'a')
                # print("dsfaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",i,j,k)
                f.write('\n' + str(complete_profile_data[i][j][k]))
                f.close()

'''
输出填充好的数据到npy文档里面
'''
def save_npy_profile_3Dimage(complete_profile_data,name_No):
    savepath = args.out_dir + str(name_No) + ".npy"
    save(savepath,complete_profile_data)




if __name__ == "__main__":
    #进行一个方向的剖面的读取（x）
    x_data_arr,x_data_label = read_profile(args.x_profile_file_path)
    print(x_data_arr,x_data_arr.shape,x_data_label,x_data_label.shape)
    complete_profile_data,num_sum = full_profile_pixel(loc_data_arr=x_data_arr)
    print("经过填充的数据大小 = ",complete_profile_data.shape)
    complete_profile_data = prepar_profile_data(loc_data_arr=x_data_arr,loc_label_arr=x_data_label,complete_profile_data=complete_profile_data,num_sum=num_sum)
    print("最终处理完的数据大小 = ", complete_profile_data.shape)
    #进行第二个方向的剖面读取（y）
    y_data_arr,y_data_label = read_profile(args.y_profile_file_path)
    print(y_data_arr,y_data_arr.shape,y_data_label,y_data_label.shape)
    complete_profile_data_y,num_sum = full_profile_pixel(loc_data_arr=y_data_arr)
    print("经过填充的数据大小 = ",complete_profile_data.shape)
    complete_profile_data = prepar_profile_data(loc_data_arr=y_data_arr,loc_label_arr=y_data_label,complete_profile_data=complete_profile_data,num_sum=num_sum)
    print("最终处理完的数据大小 = ", complete_profile_data.shape)

    #进行第三个方向剖面的读取（z）
    z_data_arr,z_data_label = read_profile(args.z_profile_file_path)
    print(z_data_arr,z_data_arr.shape,z_data_label,z_data_label.shape)
    complete_profile_data_z,num_sum = full_profile_pixel(loc_data_arr=z_data_arr)
    print("经过填充的数据大小 = ",complete_profile_data.shape)
    complete_profile_data = prepar_profile_data(loc_data_arr=z_data_arr,loc_label_arr=z_data_label,complete_profile_data=complete_profile_data,num_sum=num_sum)
    print("最终处理完的数据大小 = ", complete_profile_data.shape)


    print('填充给完毕后的含剖面训练准备数据：', complete_profile_data.shape)
    #save_txt_profile_3Dimage(complete_profile_data,"3")
    save_npy_profile_3Dimage(complete_profile_data,"1")
