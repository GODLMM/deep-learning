import tensorflow as tf 
import numpy as np
def PositionEmbedding(inputs,max_len=500,dropout=0,name='Position_Embedding'):
    """
    Position Embedding,给与序列位置信息,在时序问题中，使用位置编码往往比不使用要好
    直观上也是这样的，原理上，在时序问题上，同样元素的不同排列往往是不一样的，位置编码使这样的信息得以利用和保护
    inputs:
        param inputs:[batch_size,seq_len,seq_dim]
        max_len:最大序列长度
        dropout:embedding后进行的dropout概率
        param name
    return:
        [batch_size,seq_len,seq_dim]
    """
    with tf.name_scope(name):
        #获取输入的维度信息
        batch_size=tf.shape(inputs)[0]
        seq_len=tf.shape(inputs)[1]
        seq_dim=inputs.get_shape()[2]

        #使用numpy进行数值计算，主要是tensor有限制，不可以直接赋值
        Embedding=np.array([[pos/np.power(10000,2*i/int(seq_dim)) for i in range(int(seq_dim))] 
                            for pos in range(max_len)])
        Embedding[:,::2]=np.sin(Embedding[:,::2])
        Embedding[:,1::2]=np.cos(Embedding[:,1::2])

        #转换成tensor,并广播成与inputs同样类型
        Embedding_tensor=tf.convert_to_tensor(Embedding,dtype=tf.float32)
        Embedding_expand=tf.expand_dims(Embedding_tensor,0)
        Embedding_tile=tf.tile(Embedding_expand,[batch_size,1,1])

        #Embedded
        Embedded=Embedding_tile[:,:seq_len,:]+inputs
        Embedded=tf.layers.dropout(Embedded,rate=dropout)
        return Embedded

def Multihead_Attention(q,k,units,heads=6,dropout=0,Mask=False,training=True,reuse=False,name='Multihead_Attention'):
    """
    关键模块，多头注意力层
    inputs:
        param Q:[None,Q_len,model_dim]
        param K:[None,K_len,model_dim]
        param V:V==K
        param units:一个scale,通常是model_dim(因为同时处理，所以不除head了)
        param heads:the num of head of Attention
        param dropout:残差连接前的dropout概率
        param future_blind:if True ,未来的信息会被屏蔽(decoder)，if False:使用未来的信息(encoder)
        param training:...
        param reuse:...
        param name:...
    returns:
        [None,Q_len,model_dim]
    """
    with tf.variable_scope(name,reuse=reuse):
        #进行transform
        Q =tf.layers.dense(q,units,use_bias=False)
        K =tf.layers.dense(k,units,use_bias=False)
        V =tf.layers.dense(k,units,use_bias=False)
    
        #进行分割
        Q_split=tf.concat(tf.split(Q,heads,axis=-1),axis=0)
        K_split=tf.concat(tf.split(K,heads,axis=-1),axis=0)
        V_split=tf.concat(tf.split(V,heads,axis=-1),axis=0)

        #Q*K^T(Attention)
        Q_K=tf.matmul(Q_split,tf.transpose(K_split,[0,2,1]))
        #scale
        scale=tf.sqrt(float(units/heads))
        Q_K=Q_K/scale

        #Masking,防止未来信息对训练和预测造成影响
        #取下三角元素，mask了未来的信息
        if Mask:
            Q_K_ones=tf.ones_like(Q_K)
            future_masks=tf.matrix_band_part(Q_K_ones,-1,0)
            paddings = tf.ones_like(Q_K)*(-2**32+1)
            Q_K = tf.where(tf.equal(future_masks, 0), paddings, Q_K)
        #进行softmax,这就是为什么上面用负无穷来进行mask,到这不就变成零了吗
        Q_K = tf.nn.softmax(Q_K)

        #dropout
        Q_K = tf.layers.dropout(Q_K, rate=dropout, training=training)

        #Attention Sum
        Q_K_V=tf.matmul(Q_K,V_split)

        #还原形状加残差连接
        Q_K_V=tf.concat(tf.split(Q_K_V,heads,axis=0),axis=-1)
        outputs=Q_K_V+q

        #layer normalization
        outputs=tf.contrib.layers.layer_norm(outputs,begin_norm_axis=-1,trainable=training)
    return outputs,Q_K

def feedforward(inputs,units=[2048,512],training=True,name='Feed_Forward',reuse=False):
    """
    利用一维卷积实现的前馈网络
    inputs:
        inputs:[None,seq_len,seq_dim]
        num_units:[scale,seq_dim]
        name:...
        reuse:...
    returns:
        [None,seq_len,seq_dim]
    """
    with tf.variable_scope(name,reuse=reuse):
        outputs=tf.layers.conv1d(inputs,filters=units[0],activation=tf.nn.relu,kernel_size=1,strides=1,name='conv1')
        outputs=tf.layers.conv1d(outputs,filters=units[1],kernel_size=1,strides=1,name='conv2')
        #残差连接和layer_normalization
        outputs+=inputs
        outputs=tf.contrib.layers.layer_norm(outputs,begin_norm_axis=-1,trainable=training)
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    """
    文中的一个小trick,模型的学习会变得不确定，不过模型表现确实不错的
    Args:
        inputs:[None,seq_len,vocab_size]
        epslion:平滑程度
    returns:
        the same shape of inputs
    """
    vocab_size = inputs.get_shape().as_list()[-1]
    return ((1-epsilon) * inputs) + (epsilon / vocab_size)
