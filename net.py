import tensorflow as tf
import numpy as np
import pandas as pan
import math
import matplotlib as plt


'''
构造可训练参数
'''
def make_var(name,shape,trainable = True):
    return tf.get_variable(name,shape,trainable=trainable)

'''
定义卷积层（第三维度的数据可以类比于视频处理的关键帧）
'''
def conv3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "conv3d",biased = False):
    input_dim = input_.get_shape()[-1]#读出输入层的维度
    with tf.variable_scope(name):
        kernal = make_var(name = 'weights',shape = [kernel_size,kernel_size,kernel_size,input_dim,output_dim]) #定义卷积块
        output = tf.nn.conv3d(input_,
                              kernal,
                              [1,stride,stride,stride,1],
                              padding=padding)      #定义卷积过程
        if biased:
            biases = make_var(name = 'biases',shape=[output_dim])  #偏差
            output = tf.nn.bias_add(output,biases) #将偏差加到value上面

    return output

'''
定义反卷积层
'''
def deconv3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "deconv3d"):
    input_dim = input_.get_shape()[-1]
    input_x = int(input_.get_shape()[1])#x,y,z表示输入层相应维度的值x:height,y:width,z:chennel
    input_y = int(input_.get_shape()[2])
    input_z = int(input_.get_shape()[3])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights',shape = [kernel_size,kernel_size,kernel_size,output_dim,input_dim])
        print("input_ = ",input_)
        output = tf.nn.conv3d_transpose(input_,
                                        kernel,
                                        [1,input_z*2,input_x*2,input_y*2,output_dim],
                                        [1,2,2,2,1],
                                        padding=padding)

    return output

'''
定义归一化函数BN层(维度问题需要检查)
'''
def batch_norm(input_,name = 'batch_norm'):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        print("控制点，检查是否为后两项的计算失误",name,input_dim)
        scale = tf.get_variable("scale",
                                [input_dim],
                                initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
        offset = tf.get_variable("offset",
                                 [input_dim],
                                 initializer=tf.constant_initializer(0.0))
        print("offset,scale = ",offset,scale)
        mean,variance = tf.nn.moments(input_,axes=[1,2,3],keep_dims=True)
        print("mean,variance = ",mean,variance)
        epsilom = 1e-5
        inv = tf.rsqrt(variance + epsilom)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        test = scale * normalized
        print("各种变量的维度","scale",scale.shape,"offset",offset.shape,"normalized",normalized.shape,"相乘的维度",test.shape,"最终output的维度",output.shape)
        return output

'''
激活层：利用leakyrelu函数激活避免梯度爆炸和消失
'''
def lrelu(x,leak = 0.2,name = 'relu'):
    return tf.maximum(x,leak * x)  #relu函数本质上就是一个取大值的函数

'''
进行生成器的输出（128的数据两层卷积的U-Net）
'''
def Generator(image_3D,gf_dim = 64,reuse = False,name = 'generator'):
    input_dim = int(image_3D.get_shape()[-1])
    dropout_rate = 0.5
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        #卷积层的输入的维度为128的话就是将卷积两次最终的一层的维度为32

        #进行下采样
        print("第一个网络开始前的维度", image_3D)
        #第一个卷积层输出：（1*32*32*32*64）
        e1 = batch_norm(conv3D(input_=image_3D, output_dim=gf_dim, kernel_size=4, stride=2, name='g_conv_e1'),
                        name='g_bn_e1')
        print("第一层网络结束后的维度", e1.shape)
        # 第二个卷积层（1*16*16*16*128）
        e2 = batch_norm(conv3D(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_conv_e2'),
                        name='g_bn_e2')
        print("第二层网络结束后的维度", e2.shape)
        # 第三个卷积层(1*8*8*8*256)
        e3 = batch_norm(conv3D(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_conv_e3'),
                        name='g_bn_e3')
        print("第三层网络结束后的维度", e3.shape)
        # 第四个卷积层（1*4*4*4*512）
        e4 = batch_norm(conv3D(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e4'),
                        name='g_bn_e4')
        print("第四层网络结束后的维度", e4.shape)
        # 第五个卷积层（1*2*2*2*512）
        e5 = batch_norm(conv3D(input_=lrelu(e4), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e5'),
                        name='g_bn_e5')
        print("第五层网络结束后的维度", e5.shape)
        # 第六个卷积层（1*1*1*1*512）
        e6 = batch_norm(conv3D(input_=lrelu(e5), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e6'),
                        name='g_bn_e6')
        print("进行最终卷积以后的输出 = ", e6.shape)


        # 下采样结束，开始进行反卷积上采样
        #进行第一次反卷积（1*2*2*2*512）
        d1 = deconv3D(input_=tf.nn.relu(e6), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)  # 随机抛去无用层
        d1 = tf.concat([batch_norm(input_=d1, name='g_bn_d1'), e5], 4)
        print("反卷积第一层", d1.shape)

        #进行第二次反卷积（1*4*4*4*512）
        d2 = deconv3D(input_=tf.nn.relu(d1), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)  # 随机扔掉一般的输出
        d2 = tf.concat([batch_norm(input_=d2, name='g_bn_d2'), e4], 4)
        print("反卷积第二层 = ", d2.shape)

        #进行第三次反卷积（1*8*8*8*256）
        d3 = deconv3D(input_=tf.nn.relu(d2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_deconv_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)  # 随机扔掉一般的输出
        d3 = tf.concat([batch_norm(input_=d3, name='g_bn_d3'), e3], 4)
        print("反卷积第二层 = ", d3.shape)

        #进行第二次反卷积（1*16*16*16*128）
        d4 = deconv3D(input_=tf.nn.relu(d3), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_deconv_d4')
        d4 = tf.nn.dropout(d4, dropout_rate)  # 随机扔掉一般的输出
        d4 = tf.concat([batch_norm(input_=d4, name='g_bn_d4'), e2], 4)
        print("反卷积第二层 = ", d4.shape)

        #进行第二次反卷积（1*32*32*32*64）
        d5 = deconv3D(input_=tf.nn.relu(d4), output_dim=gf_dim, kernel_size=4, stride=2, name='g_deconv_d5')
        d5 = tf.nn.dropout(d5, dropout_rate)  # 随机扔掉一般的输出
        d5 = tf.concat([batch_norm(input_=d5, name='g_bn_d5'), e1], 4)
        print("反卷积第二层 = ", d5.shape)

        #反卷积最后一层（1*64*64*64*3）
        d_final = deconv3D(input_=tf.nn.relu(d5),output_dim=input_dim,kernel_size=4,stride=2,name='g_deconv_d6')
        print("最终反卷积的结果 = ",d_final)
        return tf.nn.tanh(d_final)




'''
定义discriminator：四层（暂定，不知道3D的生成精准度比2D高多少,层数不能太深避免训练失衡）
'''
#参数之中的image_3D为输入的原数据（train_data），targets是目标数据(train_label/gen_label)
def Discriminator(image_3D,targets,df_dim = 64,reuse = False,name = 'discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_input = tf.concat([image_3D,targets],4)#不是很明白为什么要进行合并，找其他的卷积神经网络看一下
        print("判别器输入的数据为 = ",dis_input.shape)
        #第一层卷积
        print("判别器第一层网络输入前的维度",dis_input)
        dis0 = lrelu(conv3D(input_=dis_input,output_dim=df_dim,kernel_size=4,stride=2,name='dis_con_0'))
        print("判别器第一层网络结束后的维度",dis0)
        #第2层卷积

        dis1 = lrelu(batch_norm(conv3D(input_=dis0,output_dim=df_dim*2,kernel_size=4,stride=2,name='dis_conv_1'),name='dis_bn_1'))
        print("判别器第二层网络后的维度", dis1)
        #第3层卷积

        dis2 = lrelu(batch_norm(conv3D(input_=dis1,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_2'),name='dis_bn_2'))
        print("第三层网络结束后的维度",dis2)
        #第4层卷积

        dis3 = lrelu(batch_norm(conv3D(input_=dis2,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_3'),name='dis_bn_3'))
        print("第三层网络结束后的维度",dis3)
        #最终层卷积

        dis_output = conv3D(input_=dis3,output_dim=1,kernel_size=4,stride=1,name='dis_conv_output')
        print("最终层网络结束后的维度",dis_output)
        #经过sigmoid层进行运算，作用为进行二分类运算，输出的结果为分类的结果
        dis_output = tf.sigmoid(dis_output)
        print("最终结果",dis_output)
        return dis_output




























