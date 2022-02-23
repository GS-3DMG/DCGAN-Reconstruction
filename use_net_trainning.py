'''
用来进行网络的使用，将定义的网络进行使用并通过loss函数进行计算
主要函数：

create_save_file(save_path,train_out_path):创建保存模型的文件夹
save_file(save_data,sess,log_dir,step):保存文件
get_trainning_result(trainning_3Dimage,gan_label,div_path,counter):将训练得到的矩阵进行输出到指定文件夹，得到训练的对比结果（先输出为.npy到指定文件夹,直接存放三维arr）
calcuate_lc_loss(gen_label,train_label,K,M):定义上下文的损失函数
clacuate_loss(train_label,gen_output,dis_output_fake,dis_output_real):计算总损失loss，并直接包含计算gen的loss
trainning_main():进行训练的主函数



'''
from __future__ import  absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import math
import os
import random
from random import shuffle
import cv2
import glob
import sys
import argparse

#网络结构组件
from net import *
#数据处理
from perpare_input_dataset import *
#剖面数据处理
from handel_Profile_data import *

'''
设定各类参数项及超参数项
'''
parser = argparse.ArgumentParser(description='')

args = parser.parse_args()#用来进行解析命令行的参数
parser.add_argument("--snapshot_dir",default='./model_save',help='path of snapshots')#模型保存路径
parser.add_argument("--out_dir",default='./train_out',help='path of training outputs')#训练后的生成文件输出路径
parser.add_argument("--out_trainningdata_dir",default='./dataset/train_3Dimagedata_out',help='path of trainning_out 3D data')#训练后的文件输出路径
parser.add_argument("--image_size", type=int, default=64, help="load image size") #网络输入的尺度
parser.add_argument("--image_size_z",type=int,default=64,help="load image size")#网络输入的z的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0015, help='initial learning rate for adam') #对生成器的学习率
parser.add_argument('--base_lr_dis', type=float, default=0.0002, help='initial learning rate for adam') #对判别器的学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')  #训练的epoch数量
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=500, help="times to write.") #训练中每过多少step保存结果
parser.add_argument("--save_pred_every", type=int, default=1000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--lamda_lc_weight", type=float, default=1.0, help="Lc lamda") #训练中Lc_Loss前的乘数
parser.add_argument("--lamda_gan_weight", type=float, default=1.0, help="GAN lamda") #训练中GAN_Loss前的乘数
parser.add_argument("--lamda_lc_weight_fortest", type=float, default=100.0, help="Lc lamda") #训练中Lc_Loss前的乘数
parser.add_argument("--train_data_format", default='.txt', help="format of training datas.") #网络训练输入的数据的格式(图片在CGAN中被当做条件)    暂定为txt文件
parser.add_argument("--train_label_format", default='.txt', help="format of training labels.") #网络训练输入的标签的格式(标签在CGAN中被当做真样本)    暂定为txt文件
parser.add_argument("--train_data_path", default='./dataset/train_3Dimage/', help="path of training datas.") #网络训练输入的图片路径
parser.add_argument("--train_label_path", default='./dataset/train_label/', help="path of training labels.") #网络训练输入的标签路径
parser.add_argument("--profile_path",default='./dataset/profile_data/',help="parh of profile data")#进行上下文loss计算相关性的时候需要的剖面的数据
parser.add_argument("--profile_data_size",default=86400,help="num of profile data")#进行上下文loss计算相关性的时候需要的剖面的数据(有关位置的信息数据)
parser.add_argument("--profile_label_size",default=86400,help="num of profile label")#进行上下文loss计算相关性的时候需要的剖面的数据（有关对应位置的标签的数据）
parser.add_argument("--profile_data_format",default=".txt",help="format of profile file")#进行上下文loss计算相关性的时候需要的剖面的数据（有关对应位置的标签的数据）


args = parser.parse_args()#解析命令行
EPS = 1e-12#用来保证log的参数不为负


'''
创建保存模型的文件夹
'''
def create_save_file(save_path,train_out_path):
    if not os.path.exists(save_path):#模型的保存路径
        os.makedirs(save_path)
    if not os.path.exists(train_out_path):#训练的输出路径（目前先定为txt形式输出）
        os.makedirs(train_out_path)

'''
保存文件
'''
def save_file(save_data,sess,log_dir,step):
    model_name = 'model' #定义前缀形式
    checkpoint_path = os.path.join(log_dir,model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_data.save(sess,checkpoint_path,step)
    print('checkpoint创建完毕已保存')



'''
得到训练的对比结果(直接存放txt文件）
'''
def get_trainning_result(trainning_3Dimage,gan_label,div_path,counter):#将训练得到的矩阵进行输出到指定文件夹

    # #进行保存为npy文件
    # trainning_3Dimage_path = div_path + "/out_trainning_3Dimage" + str(counter) + ".npy"
    # gan_label_path = div_path + "/gen_label" + str(counter) + ".npy"
    # np.save(trainning_3Dimage_path,trainning_3Dimage)
    # np.save(gan_label_path,gan_label)


    #进行保存为txt文件
    savepath = div_path + "/out_trainning_3Dimage" + str(counter) + ".txt"
    gan_label_path = div_path + "/gen_label" + str(counter) + ".txt"

    #进行归一化矩阵还原
    print("输出的生成矩阵的大小 = ",trainning_3Dimage.shape,gan_label.shape,type(gan_label))
    trainning_3Dimage = (trainning_3Dimage + 1.) * 127.5
    gan_label = (gan_label + 1.) * 127.5

    arr = trainning_3Dimage
    arr_gan = gan_label
    #进行trainning3Dimage的保存
    # num = arr.shape[0] * arr.shape[1] * arr.shape[2]
    # print("节点总数",num)
    f = open(savepath, 'a')
    f.write(str(arr.shape[0]) + '\t' + str(arr.shape[1]) + '\t' + str(arr.shape[2]) + '\n')
    f.write(str(1) + '\n')
    f.write(str("v"))
    f.close()
    f = open(savepath, 'a')
    for i in range(0,arr.shape[0]):
        for j in range(0,arr.shape[1]):
            for k in range(0,arr.shape[2]):
                #value = abs(arr[0][i][j][k])
                value = arr[i][j][k]
                #print("其中的值的大小",value)
                if value >= 125:
                    value = 255
                else:
                    value = 0

                f.write('\n' + str(value))
    f.close()

    #进行GAN生成结果的保存
    #num = arr_gan.shape[0] * arr_gan.shape[1] * arr_gan.shape[2]
    #print("节点总数",num)
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


'''
定义第一层的l1_loss
'''
def l1_loss(src,dst):
    return tf.reduce_mean(tf.abs(src - dst))

'''
计算loss
'''
def clacuate_loss(train_label,gen_output,dis_output_fake,dis_output_real,image3D_data,image3D_label):
    gen_loss_GAN = tf.reduce_mean(-tf.log(dis_output_fake + EPS))
    gen_loss_lcloss = calcuate_lc_loss(gen_output,image3D_data=image3D_data,image3D_label=image3D_label)
    gen_loss = gen_loss_GAN * args.lamda_gan_weight + gen_loss_lcloss * args.lamda_lc_weight
    dis_loss = tf.reduce_mean(-(tf.log(dis_output_real + EPS) + tf.log(1 - dis_output_fake + EPS)))
    return gen_loss,dis_loss

#进行实验的lc_loss
def cal_loss_test(gen_output,dis_output_fake,dis_output_real,train_data):
    #最简单的方法，直接相减求值
    lc_loss = l1_loss(gen_output,train_data)
    #进行余弦相似度的求解
    pooled_len_1 = tf.sqrt(tf.reduce_sum(gen_output * gen_output, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(train_data * train_data, 1))
    pooled_mul_12 = tf.reduce_sum(gen_output * train_data, 1)
    score = tf.reduce_mean(tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores"))

    score = 1 - score
    #进行范式求解forbenius范数
    differ = gen_output - train_data
    dist = tf.norm(differ,ord=2)
    len1 = tf.norm(gen_output,ord=2)
    len2 = tf.norm(train_data,ord=2)
    denom = (len1 + len2)/2
    similar = 1-(dist / denom)
    sim = tf.reduce_mean(similar)

    return lc_loss,score,sim




'''
进行训练的主函数
'''

def trainning_main():
    tf.set_random_seed(args.random_seed)#初始化随机数
    create_save_file(args.snapshot_dir,args.out_dir)#创建参数文件夹
    train_data_list = glob.glob(os.path.join(args.train_data_path,'*'))#训练输入的路径列表、

    #进行三维数据的读取和tf占位
    train_data = tf.placeholder(tf.float32,
                   shape=[1,args.image_size,args.image_size,args.image_size_z,3],
                   name='train_data')#输入训练图像
    train_label = tf.placeholder(tf.float32,
                   shape=[1,args.image_size,args.image_size,args.image_size_z,3],
                   name='train_label')#输入训练图像标签

    #生成器输出
    gen_output = Generator(image_3D=train_data)
    #判别器的判别结果
    dis_output_real = Discriminator(image_3D=train_data,targets=train_label,df_dim=64,reuse=False,name="discriminator")#返回真实标签结果
    print("*"*50)
    print("gen_output = ",gen_output.shape)
    print("train_data = ",train_data.shape)
    print("dis_output_real = ",dis_output_real)
    dis_output_fake = Discriminator(image_3D=train_data,targets=gen_output,df_dim=64,reuse=True,name="discriminator")#返回生成标签的对比结果、
    print("dis_output_fake = ", dis_output_fake)
    print("*" * 50)

    #计算loss
    gen_loss,dis_loss = clacuate_loss(train_label,gen_output,dis_output_fake,dis_output_real,image3D_data=train_data,image3D_label=train_label)#后面两相参数是为lcloss准备的现在先用这个代替一下
    print("@" * 100)
    print("计算出来的loss值 = ",gen_loss,dis_loss)
    print("@" * 100)
    gen_loss_sum = tf.summary.scalar('gen_loss',gen_loss)#显示标量信息
    dis_loss_sum = tf.summary.scalar('dis_loss',dis_loss)
    summary_write = tf.summary.FileWriter(args.snapshot_dir,graph=tf.get_default_graph())#记录日志



    gen_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]   #载入已经经过训练的模型数据
    dis_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]


    #进行梯度训练
    g_optim = tf.train.AdamOptimizer(args.base_lr,beta1=args.beta1)
    d_optim = tf.train.AdamOptimizer(args.base_lr_dis,beta1=args.beta1)
    g_grads_vars = g_optim.compute_gradients(gen_loss,var_list=gen_vars)#计算生成器训练的梯度
    g_train = g_optim.apply_gradients(g_grads_vars)#更新训练参数
    d_grads_vars = d_optim.compute_gradients(dis_loss,var_list=dis_vars)#计算判别器的梯度
    d_train = d_optim.apply_gradients(d_grads_vars)

    train_op = tf.group(d_train,g_train)#对多个操作进行分组



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True#设定显存不超量
    sess = tf.Session(config=config)#建立会话
    init = tf.global_variables_initializer()#新建初始化信息
    sess.run(init)#初始化参数




    saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=50)#保存模型


    counter = 0#记录训练步数

    for epoch in range(args.epoch):
        shuffle(train_data_list)
        for step in range(len(train_data_list)):
            counter += 1
            #进行训练数据的读取
            data_name,_ = os.path.splitext(os.path.basename(train_data_list[step]))
            data_resize,label_resize = handle_3Ddata(trainning_data_path=args.train_data_path,
                                                     label_data_path=args.train_label_path,
                                                     trainning_data_name=data_name,
                                                     training_data_format=args.train_data_format,
                                                     label_data_format=args.train_label_format,
                                                     standard_x=args.image_size,
                                                     standard_y=args.image_size,
                                                     standard_z=args.image_size_z)
            #进行剖面数据的读取
            #profile_data,profile_label = read_profile(filepath=args.profile_path + data_name + args.profile_data_format)
            #进行维度填充

            print("data_resize = ",data_resize.shape)
            banch_data = np.expand_dims(np.array(data_resize).astype(np.float32),axis=0)
            banch_label = np.expand_dims(np.array(label_resize).astype(np.float32),axis=0)

            #feed_dict = {train_data:banch_data,train_label:banch_label,train_profile_data:profile_data,train_profile_label:profile_label}#进行项占位项的填补
            feed_dict = {train_data: banch_data, train_label: banch_label}  # 进行项占位项的填补
            # print("输出的形状检查train_data",train_data.shape)
            # print("输出检查banch_data",banch_data.shape)
            #计算每个生成器中的gen和抵受的loss
            gen_loss_value,dis_loss_value,_ = sess.run([gen_loss,dis_loss,train_op],
                                                       feed_dict=feed_dict)
            # print("()"*100)
            # print("计算生成的最终loss结果",gen_loss_value,dis_loss_value)
            # print("()" * 100)

            if counter % args.save_pred_every == 0:
                save_file(saver,sess,args.snapshot_dir,counter)
            if counter % args.summary_pred_every == 0:
                gen_loss_sum_value,dis_loss_sum_value = sess.run([gen_loss_sum,dis_loss_sum],
                                                                 feed_dict=feed_dict)
                summary_write.add_summary(gen_loss_sum_value,counter)
                summary_write.add_summary(dis_loss_sum_value,counter)
            if counter % args.write_pred_every == 0:
                gen_label_value = sess.run(gen_output,feed_dict=feed_dict)
                #进行训练数据的输出和保存
                label_resize = label_resize[:,:,:,-1]
                gen_label_value = gen_label_value[:,:,:,:,-1]
                get_trainning_result(label_resize,gen_label_value,args.out_dir,counter)

            print('epoch {:d} step {:d} \t gen_loss = {:.3f}, dis_loss = {:.3f}'.format(epoch, step, gen_loss_value,
                                                                                        dis_loss_value))






if __name__ == "__main__":
    #print(args.out_trainningdata_dir)
    trainning_main()
























