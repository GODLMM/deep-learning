"""
coder：GODMing
mail：1415756961@qq.com
paper:Dynamic Routing Between Capsule
"""


import tensorflow as tf 

def squash(s,axis=-1,epsilon=1e-7,name='squash'):
    """
    非线性压缩函数，作为capsule的激活函数
    inputs:
        param s:输入
        param axis:需要计算范数的维度，默认最后一维
        param epsilon:防止除法发生溢出
        param name:节点名
    return:
        the vector with the same shape of s
    """
    with tf.name_scope(name):
        squash_norm=tf.reduce_sum(tf.square(s),axis=axis,keep_dims=True)
        safe_norm=tf.sqrt(squash_norm+epsilon)
        squash_factor=squash_norm/(1+squash_norm)
        temp=s/safe_norm
        return squash_factor*temp

def Dynamic_routing(inputs,routings=1,name='Dynamic_routing'):
    """
    动态路由算法(不依赖于bp算法的更新系数，一个动态规划式的算法)，动态决定输入向量的权重，有点像向量版的动态池化层(我认为)
    inputs:
        param inputs:[None,inputs_units,outputs_units,outputs_dims,1]
        param routing:路由更新循环数
        param name:节点名
    return:
        v:[None,1,outputs_units,outputs_dims,1]
    """
    with tf.name_scope(name):
        #b:[None,inputs_units,outputs_units,1,1]
        b=tf.expand_dims(tf.zeros_like(inputs[:,:,:,1]),-1)
        
        for i in range(routings):
            #c:[None,inputs_units,outputs_units,1,1]
            c=tf.nn.softmax(b,dim=2)

            #s:[None,1,outputs_units,outputs_dims,1]
            s=tf.reduce_sum(c*inputs,axis=1,keep_dims=True)

            #v:[None,1,outputs_units,outputs_dims,1]
            v=squash(s,axis=-2)

            if (i+1)<routings:
                #v_tile:[None,inputs_units,outputs_units,outputs_dims,1]
                v_tile=tf.tile(v,[1,inputs.get_shape()[1],1,1,1])

                #update b
                b=b+tf.matmul(inputs,v_tile,transpose_a=True)
    return v

def CapsuleLayer(inputs,outputs_units,outputs_dims,routings=1,reuse=False,name='Capsule'):
    """
    胶囊层
    inputs:
        param inputs:[None,inputs_units,inputs_dims]
        param outputs_units:输出胶囊数，其实和原来的神经元数差不多
        param outputs_dims:单个胶囊的维度
    return:
        outputs:[None,outputs_units,outputs_dims]
    """
    with tf.variable_scope(name,reuse=reuse):
        batch_size=tf.shape(inputs)[0]
        _,inputs_units,inputs_dims=inputs.get_shape()
        #[None,inputs_units,inputs_dims,1]
        inputs_expand=tf.expand_dims(inputs,-1)

        #[None,inputs_units,1,inputs_dims,1]
        inputs_expand=tf.expand_dims(inputs_expand,2)

        #[None,inputs_units,outputs_units,inputs_dims,1]
        inputs_tile=tf.tile(inputs_expand,[1,1,outputs_units,1,1])

        #[1,inputs_units,outputs_units,outputs_dims,inputs_dims]
        weights=tf.get_variable(name='Weights',shape=[1,inputs_units,outputs_units,outputs_dims,inputs_dims])
        
        #[None,inputs_units,outputs_units,outputs_dims,inputs_dims]
        weights_tile=tf.tile(weights,[batch_size,1,1,1,1],name='Weights_tile')
        
        #[None,inputs_units,outputs_units,outputs_dims,1]
        routing_inputs=tf.matmul(weights_tile,inputs_tile,name='routing_inputs')

        #dynamic_routing
        #[None,1,outputs_units,outputs_dims,1]
        routing_outputs=Dynamic_routing(routing_inputs,routings=routings)
        
        #[None,outputs_units,outputs_dims]
        outputs=tf.reshape(routing_outputs,[-1,outputs_units,outputs_dims])
        return outputs

def make_norm(inputs,axis=-1,keep_dims=False,name='make_dim'):
    """
    对最后输出进行二范数处理，估算分类概率
    inputs:
        param inputs:[None,inputs_units,inputs_dims]
        param axis:计算模长的维度
        param keep_dims:保持原来的维度大小
        param name:节点名
    outputs:
        默认情况下：
        [None,inputs_units]
    """
    with tf.name_scope(name):
        inputs_norm=tf.reduce_sum(tf.square(inputs),axis=axis,keep_dims=keep_dims)
        return tf.sqrt(inputs_norm)


def Margin_loss(inputs,labels,outputs_units,
                m_plus=0.9,m_minus=0.1,lambda_=0.5,name='Margin_loss'):
    """
    论文中使用的边界损失
    inputs:
        param inputs:[None,outputs_units]
        param labels:[None]
        param output_units:类别数
        param m_plus:论文也没解释，姑且认为这是一个软边界吧
        param m_minus:同样
        param lambda_:错误分类的损失权重
    """
    with tf.name_scope(name):
        #构建权重矩阵
        T=tf.one_hot(labels,depth=outputs_units)
        loss1=T*(tf.square(tf.maximum(0.,m_plus-inputs)))
        loss2=(1-T)*(tf.square(tf.maximum(0.,inputs-m_minus)))
        total_loss=loss1+lambda_*loss2
        return tf.reduce_mean(total_loss)






    

        
