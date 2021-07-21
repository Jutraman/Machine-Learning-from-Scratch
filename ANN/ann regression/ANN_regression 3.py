# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:02:48 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 10:02:19 2020

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


log_dir="D:/S1-2020.9/machine learning/data/ccpp_log"   # 保存训练日志的目录，给TensorBoard使用
test_writer=tf.summary.FileWriter(log_dir+'/test')
saver=tf.train.Saver()

def add_layer(inputs,in_size,out_size,activation_fun=None):
    Ws=tf.Variable(tf.random_normal([in_size,out_size]))
    bs=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_bs=tf.matmul(inputs,Ws)+bs
    if activation_fun is None:
        outputs=Wx_plus_bs
    else:
        outputs=activation_fun(Wx_plus_bs)
    return outputs

def normalregulation(rowdata):
    return (rowdata-np.mean(rowdata,axis=0))/np.std(rowdata,axis=0)

data=np.loadtxt(r"D:\S1-2020.9\machine learning\data\cycle power\CCPP\CCPP\data.csv",delimiter=",")


np.random.shuffle(data)



xtrain1=data[0:5741,0:4]
y_train=data[0:5741,4:]
x_test=data[5741:9568,0:4]
y_test=data[5741:9568,4:]
y=data[0:9568,4:]
x=data[0:9568,0:4]



x_ann=normalregulation(xtrain1)
y_ann=normalregulation(y_train)
testannx=normalregulation(x_test)
testanny=normalregulation(y_test)

learning_rate = 0.00141
number_epochs = 4000
mse = [None]*int(number_epochs/20)

sumloss=tf.Variable(0.,name='sum')
lossvalue=tf.Variable(0.)



xs=tf.placeholder(tf.float32,[None,4])
ys=tf.placeholder(tf.float32,[None,1])

n_hidden1=add_layer(xs,4,10,None)
n_hidden2=add_layer(n_hidden1, 10, 10,tf.nn.sigmoid)
preditdata=add_layer(n_hidden2,10,1,None) # neu network

loss=tf.sqrt(tf.reduce_mean((tf.keras.losses.mean_squared_error(ys, preditdata))))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) 

init=tf.initialize_all_variables() 

def get_accuracy(preditdata, ys):
    accuracy = tf.sqrt(tf.reduce_mean((tf.keras.losses.mean_squared_error(ys, preditdata))))
    tf.summary.scalar('accuracy',accuracy)
    return accuracy

def train(sess, init, train_step, number_epochs, x_train, y_t):

    sess.run(init) 
    for i in range(number_epochs):
        sess.run(train_step, feed_dict={xs:x_train, ys:y_t})
#        if i % 50 == 0:
#            mse1 = get_accuracy(preditdata, y_t)
#            mse[int(i/50)] = mse1.eval({xs:x_train, ys:y_t})
        if i % (number_epochs // 10) == 0:
                mse1 = get_accuracy(preditdata, y_t)
                print("LOSS:", mse1.eval({xs:x_train, ys:y_t}))
    

    mse1 = get_accuracy(preditdata, y_t)
    final_acc = mse1.eval({xs:x_train, ys:y_t})

    return final_acc

def test(sess, init, train_step, number_epochs, x_train, y_t):

    sess.run(init) 
    for i in range(number_epochs):
        sess.run(train_step, feed_dict={xs:x_train, ys:y_t})
        if i % 20 == 0:
            mse1 = get_accuracy(preditdata, y_t)
            mse[int(i/20)] = mse1.eval({xs:x_train, ys:y_t})
            if i % (number_epochs // 10) == 0:
                    mse1 = get_accuracy(preditdata, y_t)
                    print("LOSS:", mse1.eval({xs:x_train, ys:y_t}))
    

    mse1 = get_accuracy(preditdata, y_t)
    final_acc = mse1.eval({xs:x_train, ys:y_t})

    return final_acc

def traintest(sess, init, train_step, number_epochs, x_test, y_test):
#    loss=[]
    testac=train(sess, init, train_step, number_epochs, x_test, y_test)
    return testac

def k_fold(k ,sess, init, train_step, number_epochs, x_ann, y_ann):
    sumt=0
    sumv=0
    for i in range(k):
        x_train, y_train, x_valid, y_valid = get_k_fold_data(x_ann, y_ann, i, k)
        train_acc = train(sess, init, train_step, number_epochs, x_train, y_train)
        mse_valid = get_accuracy(preditdata, y_valid)
        valid_acc = mse_valid.eval({xs: x_valid, ys: y_valid})
        print("Fold", i,"train loss:", train_acc, "valid loss:", valid_acc)
        sumt += train_acc
        sumv += valid_acc
    return sumt/k,sumv/k

def get_k_fold_data(x_data, y_data, i, k):
    fold_size = x_data.shape[0] // k
    x_train, y_train = None, None
    x_valid, y_valid = None, None

    for j in range(k):

        index = slice(j * fold_size, (j + 1) * fold_size)
        x_part = x_data[index, :]
        y_part = y_data[index, :]

        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = np.vstack((x_train, x_part))
            y_train = np.vstack((y_train, y_part))

    return x_train, y_train, x_valid, y_valid


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ls = [None]*75
    k = 10
    taccu,vaccu=k_fold(k ,sess, init, train_step, number_epochs, x_ann, y_ann)
    print("eventually train loss:",taccu,"validation loss",vaccu)   
    loss=test(sess, init, train_step, number_epochs, testannx, testanny)
    print("Test data loss:",loss)
    output=preditdata.eval({xs: x_ann, ys: y_ann})
    
    plt.plot( y_ann[:75], 'r',  output[:75], 'b', ls, )
    plt.ylabel('output')
    plt.show()
    plt.xticks(range(60,10))
    plt.plot(mse,'b')
    plt.show()
    
    
    saver = tf.train.Saver()
    saver.save(sess,r"D:\S1-2020.9\machine learning\data\ccpp_model", global_step=1000)
   