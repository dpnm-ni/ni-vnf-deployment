import tensorflow as tf
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import metrics
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Dense, Softmax, Dropout, Flatten, Reshape, BatchNormalization, Activation, concatenate
from keras.models import load_model, Model
from tensorflow.keras.models import load_model, model_from_json
from spektral.layers import ECCConv
from spektral.utils import label_to_one_hot

tf.test.is_gpu_available()
print(tf.__version__)
scaler = StandardScaler()

learning_rate = (1e-3)  #1e-3      
epochs = 2000               
batch_size = 100000    
es_patience = 300     
test_split = 0.1 
val_split = 0.1 

model_type="GNN"#"FNN"


#SFC request Data
X1=[]			# traffics
#Network Topology & Labeling Output
X2=[]			# Network Node Feature
A2=[]			# Network Topology
E2=[]			# Network Edge Feature
# Labeling Output
Y=[]
	
def training_data_load(path):

    global X1, X2, A2, E2, Y
    
    trafficsPath = 'dataset/'+path+'/traffics'
    node_matrixPath = 'dataset/'+path+'/node_matrix'
    edge_matrixPath = 'dataset/'+path+'/edge_matrix'
    adjacency_matrixPath = 'dataset/'+path+'/adjacency_matrix'
    deploymentsPath = 'dataset/'+path+'/deployments'
    
    with open(trafficsPath, 'rb') as fr:
        try:
            while True:
                u = pickle._Unpickler(fr)
                u.encoding='latin1'
                X1 = np.asarray(u.load())
        except EOFError:
            pass


    with open(node_matrixPath, 'rb') as fr:
        try:
            while True:
                u = pickle._Unpickler(fr)
                u.encoding='latin1'
                X2 = np.asarray(u.load())
        except EOFError:
            pass

    with open(edge_matrixPath, 'rb') as fr:
        try:
            while True:
                u = pickle._Unpickler(fr)
                u.encoding='latin1'
                E2 = np.asarray(u.load())
        except EOFError:
            pass

    with open(adjacency_matrixPath, 'rb') as fr:
        try:
            while True:
                u = pickle._Unpickler(fr)
                u.encoding='latin1'
                A2 = np.asarray(u.load())
        except EOFError:
            pass

    try :
        with open(deploymentsPath, 'rb') as fr:
            try:
                while True:
                    u = pickle._Unpickler(fr)
                    u.encoding='latin1'
                    Y = np.asarray(u.load())
                    Y = Y.astype(np.float64)
            except EOFError:
                pass
    except :
        return     
           

    return 

'''
def c_cross_entropy(y_true, y_pred):
    a = tf.convert_to_tensor([0],dtype=tf.float32)
    for l in range(0,Y.shape[1]):                           ##Group for Number of Servers
        batch_mask = K.equal((y_true[:,l,0,0]),4)    ##Figure out which batch has padding signal or switch signal as "4"
        batch_mask = ~batch_mask                     ##Reversed batch => batch that were not padded
        for i in range(0,Y.shape[2]):                                                        ##Group for Type of VNFs
            t1 = tf.boolean_mask(y_true[:,l,i], batch_mask)                         ##Extract y_ture[:,l,i] which were not padded (ex.. only 700 is extracted from 1000 batch)
            g = tf.reduce_sum(t1, axis=0)            ##Sum for batch => [200,300,200]
            h = tf.reduce_sum(g, axis=-1)            ##Sum for All => [700]
            g = (g/h) #* [1,0.01,1]                              ##[0.29, 0.43, 0.29]
            w = tf.multiply(y_true[:,l,i], g)        ##y_true * [0.29, 0.43, 0.29]
            w = tf.reduce_sum(w, axis = -1)          ##[[0.29],[0.29],...[0],...[0.43]]
            w = w + K.cast(~batch_mask, 'float32')   ##When w has 0 value, then it evokes Nan error. Thus, when 0 existed, should add 1.(1 does not act any role in dividing)
            p = K.cast(batch_mask,'float32') * K.categorical_crossentropy(y_true[:,l,i], y_pred[:,l,i]) ##Batch * cross entropy
            q = tf.cond((h > 0.0), lambda : p/w, lambda:p)                          ##If all graph is padded then just use p. if not, then devide p by w (labeling balance)
            a = a+q                                  ##update a
    return a
'''


#tf.argmax is not differentiable.. should use softargmax.

def custom_loss(y_true, y_pred):
    cumulative_c_c_entropy = tf.convert_to_tensor([0],dtype=tf.float32)
    for i in range(Y.shape[-3]):
        for j in range(Y.shape[-2]):
            cumulative_c_c_entropy += K.categorical_crossentropy(y_true[:,i,j], y_pred[:,i,j])
    
    '''
    argmax = tf.constant([0, 1, 2], dtype= tf.float32)
    tf.print("ypred : ", y_pred)
    argmax_vnfs_pred = argmax * y_pred
    tf.print("multipled : ", argmax_vnfs_pred) 
    argmax_vnfs_pred = tf.reduce_sum(argmax_vnfs_pred, axis = -1)
    num_vnfs_pred = tf.reduce_sum(argmax_vnfs_pred, axis = (1,2))
    
    argmax_vnfs_true = argmax * y_true
    argmax_vnfs_true = tf.reduce_sum(argmax_vnfs_true, axis = -1)    
    num_vnfs_true = tf.reduce_sum(argmax_vnfs_true, axis = (1,2))
    loss = K.cast(num_vnfs_pred - num_vnfs_true, tf.float32)
    '''
    
    loss = cumulative_c_c_entropy
    loss = K.cast(loss, tf.float32)
    return loss


# Metric Function
# Show Metric at each iteration (If you want final metric, then you should check "Evaluating model Result")
def custom_metric(y_true, y_pred):
    ans_suc,ans_all = 0,0
    for l in range(0,Y.shape[1]):
        batch_mask = K.equal((y_true[:,l,0,0]),4)
        batch_mask = ~batch_mask
        sum_batch = tf.reduce_sum(K.cast(batch_mask,'int32'))
        for i in range(0,Y.shape[2]):
            p=K.cast(K.equal(K.argmax(y_true[:,l,i,:],axis=-1),K.argmax(y_pred[:,l,i,:],axis=-1)),K.floatx())* K.cast(batch_mask,'float32')
            ans_suc = ans_suc + tf.reduce_sum(p,axis=-1)
            ans_all = ans_all + K.cast(sum_batch,'float32')            
    return ans_suc/ans_all
    

def num_vnfs_metric(y_true, y_pred):

    argmax_vnfs_pred = tf.argmax(y_pred, axis= -1)
    num_vnfs_pred = tf.reduce_sum(argmax_vnfs_pred, axis = (1,2))
    
    argmax_vnfs_true = tf.argmax(y_true, axis= -1)
    num_vnfs_true = tf.reduce_sum(argmax_vnfs_true, axis = (1,2))    

    
    return (num_vnfs_pred - num_vnfs_true)



def run_mygnn(is_trained=True, is_router_included=True, env="simulation"):

    training_data_load(env)

    global X1,X2,A2,E2,Y
    
    print(X1[0:100])
    print(X2[0:100])
    #print(X2)      
    #print(A2)
    #print(E2)
    #print(Y)

    model_env = env
    
    non_zero_one_indices = (np.where((X1[0] != 0) & (X1[0] != 1))[0]).tolist()    
    scaler.fit(X1[:, non_zero_one_indices])
    X1[:, non_zero_one_indices] = scaler.fit_transform(X1[:,non_zero_one_indices])      
      
    X2_shape = X2.shape
    scaling_X2 = np.reshape(X2, (X2_shape[0] * X2_shape[1], X2_shape[2]))
    scaler.fit(scaling_X2)
    scaling_X2 = scaler.fit_transform(scaling_X2)
    X2 = np.reshape(scaling_X2, (X2_shape[0], X2_shape[1], X2_shape[2]))    
      
    E2_shape = E2.shape
    scaling_E2  = np.reshape(E2, (E2_shape[0] * E2_shape[1] * E2_shape[2], E2_shape[3]))
    scaler.fit(scaling_E2)
    scaling_E2 = scaler.fit_transform(scaling_E2)
    E2 = np.reshape(scaling_E2, (E2_shape[0], E2_shape[1], E2_shape[2], E2_shape[3]))
    
    numberNodes = 0
    numberVNFsType = 0
    
    
    
    if is_router_included == False:
        Y = Y[:, :9, 1:5, :]
        
        
    if is_trained == True:
        model = tf.keras.models.load_model('save/deployment.h5', custom_objects={'ECCConv': ECCConv, 'custom_loss': custom_loss, 'custom_metric': custom_metric, 'num_vnfs_metric': num_vnfs_metric})
        prediction = model.predict([X1, X2, A2, E2])
        
        if model_env == "simulation":
            eval_results = model.evaluate([X1, X2, A2, E2],Y, batch_size=1024)
            print('Done.\nTest loss: {}\nTest accuracy: {}\n'.format(*eval_results))
            return (eval_results[-1], prediction)
        else :
            eval_results = model.evaluate([X1, X2, A2, E2],Y, batch_size=1024)
            print('Done.\nTest loss: {}\nTest accuracy: {}\n'.format(*eval_results))
            print(prediction)
            print(np.argmax(prediction[0], axis=-1))
            return prediction

 
    X1_in = Input(shape=X1.shape[1:])

    X2_in = Input(shape=X2.shape[1:])
    A2_in = Input(shape=A2.shape[1:])
    E2_in = Input(shape=E2.shape[1:])
    
    
    X1_train, X1_test, \
    X2_train, X2_test, \
    A2_train, A2_test, \
    E2_train, E2_test, \
    Y_train, Y_test = train_test_split(X1, X2, A2, E2, Y, test_size=test_split, shuffle = True, random_state = 3)
    
    topo_EdgeGNN = ECCConv(100)([X2_in, A2_in, E2_in])
    topo_EdgeGNN = Flatten()(topo_EdgeGNN)  #use_bias=False, BatchNormalization(), Activation('relu'), Dropout(0.5)
    topo_EdgeGNN = Flatten()(X2_in)
    
    traffic_FNN = (X1_in)
    traffic_FNN = Dense(500)(traffic_FNN)
    traffic_FNN = BatchNormalization()(traffic_FNN)
    traffic_FNN = Activation('relu')(traffic_FNN)
    traffic_FNN = Dropout(0.5)(traffic_FNN)   

    network_CCT = concatenate([traffic_FNN,topo_EdgeGNN],axis=-1)
    
    network_FCN = Dense(500)(network_CCT)
    network_FCN = BatchNormalization()(network_FCN)
    network_FCN = Activation('relu')(network_FCN)
    network_FCN = Dropout(0.5)(network_FCN)    
    

    network_FCN = Dense(300)(network_FCN)    
    network_FCN = BatchNormalization()(network_FCN)
    network_FCN = Activation('relu')(network_FCN)
    network_FCN = Dropout(0.5)(network_FCN)   
 
    
    output = Dense((Y.shape[1]) * Y.shape[2] * Y.shape[3])(network_FCN)
    output = Reshape((Y.shape[1] , Y.shape[2] , Y.shape[3]))(output)
    output = Softmax(axis=-1)(output)
    
    model = Model(inputs=[X1_in, X2_in, A2_in, E2_in], outputs=output)
    optimizer = Adam(lr=learning_rate)
    
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[custom_metric, num_vnfs_metric])

    model.summary()

    # Callbacks
    es_callback = EarlyStopping(monitor='val_loss', patience=es_patience)

    # Train model
    model.fit([X1_train, X2_train, A2_train, E2_train],
              Y_train,
              batch_size=batch_size,
              validation_split=val_split,
              epochs=epochs,
              callbacks=[es_callback] )

    #Save Model
    if is_trained == False:
        model.save("save/deployment.h5")

    # Evaluate model
    print('Evaluating model.')
    eval_results = model.evaluate([X1_test, X2_test, A2_test, E2_test],Y_test, batch_size=1024)


    print('Done.\nTest loss: {}\nTest accuracy: {}\n'.format(*eval_results))
          
          
    return eval_results[-1]



#run_mygnn(is_trained=False, is_router_included=True, env="val")
#run_mygnn(is_trained=True, is_router_included=True, env="simulation")
#run_mygnn(is_trained=True, is_router_included=True, env="testbed")





































