
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# create random training data again
Nclass = 500
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

N = len(Y)
# turn Y into an indicator matrix for training
T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X,W1,b1,W2,b2):
    # sigmoind (X times W1+b1) times W2+b2
    Z=tf.nn.sigmoid(tf.matmul(X,W1)+b1)
    Y=tf.matmul(Z,W2)+b2
    return Y #not softmax yet I guess

#placeholders for data
tfX=tf.placeholder(tf.float32,[None,D]) #do not know number of rows  yet
tfY=tf.placeholder(tf.float32,[None,K]) #do not know number of rows  yet
#create initial random weights
W1=init_weights([D,M])
b1=init_weights([M])
W2=init_weights([M,K])
b2=init_weights([K])

pyx=forward(tfX,W1,b1,W2,b2)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY,logits=pyx))

#training function
train_op=tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op=tf.argmax(pyx,1)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_op,feed_dict={tfX:X,tfY:T})
    pred=sess.run(predict_op,feed_dict={tfX:X,tfY:T})
    if i% 10==0:
        print(np.mean(Y==pred))#print accuracy

    

  