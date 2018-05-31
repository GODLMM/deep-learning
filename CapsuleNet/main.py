import tensorflow as tf 
from models import *
import tensorlayer as tl 
import numpy as np 
import os
import time

#定义超参数
epochs=20
batch_size=32
learning_rate=0.001
image_size=28
channel=1
outputs_units=10
outputs_dims=16
use_true=True#是否用真是标签重构，否则用预测标签
alpha=0.0005#重构损失的权重，有点像一个正则化项，防止过拟合

#读取mnist数据
x_train,y_train,x_val,y_val,x_test,y_test=tl.files.load_mnist_dataset(shape=(-1,28,28,1))

#构建网络
X_train=tf.placeholder(tf.float32,shape=[None,image_size,image_size,channel],name='inputs')
y=tf.placeholder(tf.int64,shape=[None],name='labels')
conv1=tf.layers.conv2d(X_train,filters=256,kernel_size=(9,9),strides=(1,1),
                        padding='valid',activation=tf.nn.relu,name='conv1')

conv2=tf.layers.conv2d(conv1,filters=256,kernel_size=(9,9),strides=(2,2),
                        padding='valid',activation=tf.nn.relu,name='conv2')

reshape=tf.reshape(conv2,[-1,32*6*6,8],name='Reshape')

#squash
squash=squash(reshape,name='Squash')

#使用capsule层
capsule=CapsuleLayer(squash,outputs_units=outputs_units,outputs_dims=outputs_dims,
                        routings=3,name='capsule')

#计算模长
norm=make_norm(capsule,name='norm')

#预测
outputs=tf.argmax(norm,axis=-1,name='outputs')

#边界损失
m_loss=Margin_loss(norm,y,outputs_units)

#重构网络
with tf.variable_scope('Decoder',reuse=False):
    if use_true:
        mask=T=tf.one_hot(y,depth=outputs_units)
    else:
        mask=T=tf.one_hot(outputs,depth=outputs_units)
    mask=tf.expand_dims(mask,-1)
    capsule_masked=capsule*mask
    decoder_inputs=tf.reshape(capsule_masked,shape=[-1,outputs_units*outputs_dims],
                                name='decoder_inputs')
    fc1=tf.layers.dense(decoder_inputs,units=512,activation=tf.nn.relu,name='FC1')
    fc2=tf.layers.dense(fc1,units=1024,activation=tf.nn.relu,name='FC2')
    fc3=tf.layers.dense(fc2,units=image_size*image_size*channel,activation=tf.nn.sigmoid,name='FC3')
    decoder_outputs=tf.reshape(fc3,[-1,image_size,image_size,channel])

#重构损失
r_loss=tf.reduce_mean(tf.square(decoder_outputs-X_train),name='R_loss')

#总损失
total_loss=m_loss+alpha*r_loss

#正确率
correct=tf.equal(y,outputs,name='correct')
acc=tf.reduce_mean(tf.cast(correct,tf.float32),name='accuracy')

#训练节点，使用了Adam
train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

init=tf.global_variables_initializer()
saver=tf.train.Saver()
sess=tf.Session()
sess.run(init)
train_writer=tf.summary.FileWriter('logdir',sess.graph)
train_writer.add_graph(sess.graph)

#saver.restore(sess,'./model/model.ckpt')

best_acc=0
for epoch in range(epochs):
    idx=x_train.shape[0]//batch_size
    start=time.time()
    all_loss=0
    for i in range(idx):
        batch_x=x_train[i*batch_size:(i+1)*batch_size]
        batch_y=y_train[i*batch_size:(i+1)*batch_size]
        feed_dict={X_train:batch_x,y:batch_y}
        val_dict={X_train:x_val[:100],y:y_val[:100]}
        sess.run(train_op,feed_dict=feed_dict)
        Total_loss,train_acc=sess.run([total_loss,acc],feed_dict=feed_dict)
        all_loss=(i*all_loss+Total_loss)/(i+1)
        if (i+1)%100==0:
            print('100 batchs finished!')
            val_acc=sess.run(acc,feed_dict=val_dict)
            val_decoder=sess.run(decoder_outputs,feed_dict=val_dict)
    
            #保存重构图片
            tl.visualize.save_images(val_decoder,[10,10],'images/val_{:03d}_{:04d}.jpg'.format(epoch,i))
            print('epoch: %d, batch_size: %d,time: %.3f,train_loss: %.4f,train_acc: %.3f,val_acc: %.3f'%(epoch,i+1,time.time()-start,all_loss,train_acc,val_acc))
            #保存模型
            if val_acc>=best_acc-0.0001:
                saver.save(sess,'./model/model.ckpt')
                best_acc=val_acc


    

