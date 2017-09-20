import numpy as np
import pandas as pd
from sklearn import manifold
import matplotlib.pyplot as plt

import pickle


train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


X1 = train.values[:,1:-1]
X2 = test.values[:,1:] # same with above

X = np.concatenate((X1,X2), axis= 0)

np.amax()

y3 = train.values[:-1]
y1 = train.values[:,-1]
y1 = np.asarray([np.log(y) for y in y1]) # log를 취해버리네

# Preprocessing

# 1: separate different variable types
cat_idx = []
float_idx = []
int_idx = []


# sort X1 according to it's type
for c in range(X1.shape[1]) :
    typeStr = type(  X1[0, c] ).__name__
    if typeStr == 'str'  :
        cat_idx.append(c)
    elif typeStr == 'float' and X1[1,c] == X1[1,c] :
        float_idx.append(c)
    elif typeStr == 'int' :
        int_idx.append(c)

# 2. encode string values into numbers
for c in cat_idx :
    uniques = list(set(X[:,c]))
    tmp_dict = dict(zip(uniques, range(len(uniques))))
    n_enc = np.array([tmp_dict[s] for s in X[:,c]]) #우와 이런 신기한 문법이 ㅎㅎ 참 간편하네
    X[:,c] = n_enc

# 3. what does an embedding of all int values look like?
print('embedding int values..')
plt.figure(1)

X_int = X[:,np.array(int_idx)]
X_int = np.float64(X_int)

#replace nan
X_int[X_int!=X_int] = 0   # 이부분이 이해가 안가네 nan을 0으로 바꾸는 부분인가?
X_int -= np.min(X_int, axis= 0) # 이부분은 generalize하는건가
X_int /= (.001+ np.max(X_int, axis=0)) # 맥스값에 min value(0.001)을 더해서 이걸로 원래 값을 나눠버린다 역시 generalization

tsne = manifold.TSNE(n_components=2, init='pca')
Y_int = tsne.fit_transform(X_int)

plt.scatter(Y_int[len(X1):,0], Y_int[len(X1):,1], marker='.', label = 'test')
sp = plt.scatter(Y_int[:len(X1),0], Y_int[:len(X1),1], c=y1, label= 'train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of int variables')
plt.savefig('t-SNE_int.png')
plt.show()




# 4: what does an embedding of all string values look like?
print('embedding string values...')
plt.figure(2)
X_str = X[:,np.array(cat_idx)]
# replace nan
X_str[X_str!=X_str] = 0


def onehot(x):
    nx=np.zeros((len(x),max(x)+1))
    for k in range(len(x)):
        nx[k,x[k]] = 1
    return nx

X_tmp = []
for c in range(X_str.shape[1]):
    X_tmp.extend(onehot(X_str[:,c]).T)
X_str = np.asarray(X_tmp).T
tsne = manifold.TSNE(n_components=2,init='pca')
Y_str = tsne.fit_transform(X_str)
#y1-=np.nanmin(y1)
#y1/=np.nanmax(y1)
plt.scatter(Y_str[len(X1):,0],Y_str[len(X1):,1],marker='.',label='test')
sp = plt.scatter(Y_str[:len(X1),0],Y_str[:len(X1),1],c=y1,label='train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of string variables')
plt.savefig('t-SNE_string.png')

# 4: what does an embedding of all int and string values look like?
print('embedding int and string values...')
plt.figure(3)
X_strint = np.concatenate((X_int,X_str),axis=1)
tsne = manifold.TSNE(n_components=2,init='pca')
Y_strint = tsne.fit_transform(X_strint)
plt.scatter(Y_strint[len(X1):,0],Y_strint[len(X1):,1],marker='.',label='test')
sp = plt.scatter(Y_strint[:len(X1),0],Y_strint[:len(X1),1],c=y1,label='train')
plt.legend(prop={'size':6})
plt.colorbar(sp)
plt.title('t-SNE embedding of int and string variables')
plt.savefig('t-SNE_intstring.png')

# center data at 0 scaled from -0.5 to +0.5 for neural networks
# -> start within the linear region of tanh activation function
X_strint-=.5
X_strint_train = X_strint[:len(X1),:]
X_strint_test = X_strint[len(X1):,:]

pickle.dumps()

from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization
from keras import regularizers

#--------------------------------------
# Neural Network
#--------------------------------------

inp = Input(shape=(X_strint.shape[1],))
D1 = Dropout(.1)(inp)
L1_normal = Dense(64, init='uniform', activation='tanh')(D1)
L1_sparse = Dense(1024, init='uniform', activation='tanh',activity_regularizer=regularizers.l1(.001))(D1)
L2_normal = Dense(1, init='uniform', activation='tanh')(L1_normal)
L2_sparse = Dense(1, init='uniform', activation='tanh')(L1_sparse)
# models that are trained to predict prices
NN1 = Model(inp,L2_normal)
NN2 = Model(inp,L2_sparse)
# models for reading out activations in hidden layers
enc1 = Model(inp,L1_normal)
enc2 = Model(inp,L1_sparse)
# compile models
NN1.compile(loss='mse', optimizer='adam', metrics=['mae'])
NN2.compile(loss='mse', optimizer='adam', metrics=['mae'])
enc1.compile(loss='mse', optimizer='adam', metrics=['mae'])
enc2.compile(loss='mse', optimizer='adam', metrics=['mae'])
# train
min_y1 = np.min(y1)
max_y1 = np.max(y1)
scaled_y1 = y1-min_y1
scaled_y1/=max_y1
scaled_y1-=.5
NN1.fit(X_strint_train[:-50,:], scaled_y1[:-50], epochs=15, batch_size=3, shuffle=True,verbose=False)
NN2.fit(X_strint_train[:-50,:], scaled_y1[:-50], epochs=15, batch_size=3, shuffle=True,verbose=False)

# get hidden layer activations
P1 = enc1.predict(X_strint)
P2 = enc2.predict(X_strint)
# get 2d embeddings
tsne = manifold.TSNE(n_components=2,init='pca')
P1_tsne = tsne.fit_transform(P1)
P2_tsne = tsne.fit_transform(P2)

P1_tsne_train = P1_tsne[:len(X1),:]
P2_tsne_train = P2_tsne[:len(X1),:]
P1_tsne_test = P1_tsne[len(X1):,:]
P2_tsne_test = P2_tsne[len(X1):,:]

plt.figure(1)
plt.scatter(P1_tsne_test[:,0],P1_tsne_test[:,1],marker='.',label='test')
sp1 = plt.scatter(P1_tsne_train[:-50,0],P1_tsne_train[:-50,1],c=y1[:-50],label='train')
plt.scatter(P1_tsne_train[-50:,0],P1_tsne_train[-50:,1],marker='^',s=55, c=y1[-50:],label='validation')
plt.colorbar(sp1)
plt.legend(prop={'size':6})
plt.title('t-SNE embedding layer1')
plt.savefig('t-SNE_layer1.png')

plt.figure(2)
plt.scatter(P2_tsne_test[:,0],P2_tsne_test[:,1],marker='.',label='test')
sp2 = plt.scatter(P2_tsne_train[:-50,0],P2_tsne_train[:-50,1],c=y1[:-50],label='train')
plt.scatter(P2_tsne_train[-50:,0],P2_tsne_train[-50:,1],marker='^',s=55, c=y1[-50:],label='validation')
plt.colorbar(sp2)
plt.legend(prop={'size':6})
plt.title('t-SNE embedding sparse layer1')
plt.savefig('t-SNE_layer1sparse.png')

