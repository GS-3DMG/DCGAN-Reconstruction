'''
进行对训练后的网络的调用，输入带训练的数据最终输出相应的经过GAN网络生成器生成的数据

'''

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import argparse
import sys
import math
import glob


from use_net_trainning import *
from net import *
from perpare_input_dataset import *

parser = argparse.ArgumentParser(description='')

parser.add_argument("--test_3Dimage_path", default='./dataset/test_3Dimage/', help="path of test datas.")  # 网络测试输入的图片路径
parser.add_argument("--test_label_path", default='./dataset/test_label/', help="path of test datas.")  # 网络测试输入的标签路径
parser.add_argument("--image_size", type=int, default=64, help="load image size")  # 网络输入的尺度
parser.add_argument("--test_input_size_x",type=int,default=64,help='input 3Dimage size for x')#网络的输入的维度
parser.add_argument("--test_input_size_y",type=int,default=64,help='input 3Dimage size for y')#网络的输入的维度
parser.add_argument("--test_input_size_z",type=int,default=64,help='input 3Dimage size for z')#网络的输入的维度
parser.add_argument("--test_3Dimage_format", default='.txt', help="format of test pictures.")  # 网络测试输入的图片的格式
parser.add_argument("--test_label_format", default='.txt', help="format of test labels.")  # 网络测试时读取的标签的格式
parser.add_argument("--snapshots", default='./model_save/', help="Path of Snapshots")  # 读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_output/', help="Output Folder")  # 保存网络测试输出图片的路径

args = parser.parse_args()  # 用来解析命令行参数

'''
#进行文件的读取
'''
def read_npy_trainning_3Dimage(filepath):
    output_arr = np.load(filepath)
    return output_arr


'''
#进行文件的保存
'''
#保存为相应的.npy文件
def save_npy_test_3Dimage(filepath,save_file):
    np.save(filepath,save_file)

#保存为相应的txt文件
def save_txt_test_3Dimage(gan_label_path,arr_gan):

    #进行归一化的展开操作
    arr_gan = (arr_gan + 1.) * 127.5

    #进行最终的输出操作
    f = open(gan_label_path, 'a')
    f.write(str(arr_gan.shape[1]) + '\t' + str(arr_gan.shape[2]) + '\t' + str(arr_gan.shape[3]) + '\n')
    f.write(str(1) + '\n')
    f.write(str("v"))
    f.close()
    f = open(gan_label_path, 'a')
    for i in range(0,arr_gan.shape[1]):
        for j in range(0,arr_gan.shape[2]):
            for k in range(0,arr_gan.shape[3]):
                #value = abs(arr_gan[0][i][j][k])
                value = arr_gan[0][i][j][k]
                print("其中的值的大小",value)
                if value >= 125:
                    value = 255
                else:
                    value = 0

                f.write('\n' + str(value))
    f.close()




#进行操作的主函数
def use_model_classify():
    #检查文件夹是否创建,并进行创建文件夹
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    #得到测试列表
    test_3Dimage_list = glob.glob(os.path.join(args.test_3Dimage_path,'*'))
    test_3Dimage = tf.placeholder(tf.float32,
                                  shape=[1,args.test_input_size_x,args.test_input_size_y,args.test_input_size_z,3],
                                  name='test_picture')

    #得到生成器的生成结果
    gan_label = Generator(image_3D=test_3Dimage,gf_dim=64,reuse=False,name='generator')
    restore_var = [v for v in tf.global_variables() if 'generator' in v.name] #进行一训练模型的重新加载


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True#设定显存不超量
    sess = tf.Session(config=config)#新建会话层

    saver = tf.train.Saver(var_list=restore_var,max_to_keep=1)#导入参数模型
    checkpoint = tf.train.latest_checkpoint(args.snapshots)#读取模型参数
    saver.restore(sess,checkpoint) #倒入模型参数

    for step in range(len(test_3Dimage_list)):
        #读取图片
        image3D_name,_ = os.path.splitext(os.path.basename(test_3Dimage_list[step]))
        data_resize, label_resize = handle_3Ddata(trainning_data_path=args.test_3Dimage_path,
                                                  label_data_path=args.test_label_path,
                                                  trainning_data_name=image3D_name,
                                                  training_data_format=args.test_3Dimage_format,
                                                  label_data_format=args.test_label_format,
                                                  standard_x=args.test_input_size_x,
                                                  standard_y=args.test_input_size_y,
                                                  standard_z=args.test_input_size_z)

        #进行第一维度的填充
        print("data_resize = ", data_resize.shape)
        banch_data = np.expand_dims(np.array(data_resize).astype(np.float32), axis=0)

        feed_dict = {test_3Dimage:banch_data}
        gen_label_value = sess.run(gan_label,feed_dict=feed_dict)#得到生成结果、
        np.set_printoptions(suppress=True)
        print("得到的最终的生成结果",gen_label_value)

        #进行训练完毕后的文件的保存
        save_path1 = args.out_dir + image3D_name + ".npy"
        save_path2 = args.out_dir + image3D_name + ".txt"
        gen_label_value = gen_label_value[:,:,:,:,-1]
        #save_npy_test_3Dimage(save_path,gen_label_value)
        save_txt_test_3Dimage(save_path2,gen_label_value)

        #输出必要的提示信息
        print('step {:d}'.format(step))



if __name__ == "__main__":
    use_model_classify()





























