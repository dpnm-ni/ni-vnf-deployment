from __future__ import print_function

import json
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import threading
import pickle
import datetime as dt
import ni_mon_client
import ni_nfvo_client
import numpy as np
from datetime import datetime
from keras.engine.saving import load_model, model_from_json
from spektral.layers import EdgeConditionedConv

from config import cfg
import time

from builtins import range
import glob
import re
import subprocess


    
with open('vnf_placement_model_cnsm.json', 'r') as f:
        model_dict = json.load(f)

model_json = json.dumps(model_dict)
model = model_from_json(model_json)
model.load_weights('vnf_placement_model_cnsm.h5')

with open('X1_sfc1_ni.pickle', 'rb') as fr:
            try:
                while True:
                    u = pickle._Unpickler(fr)
                    u.encoding='latin1'
                    p = u.load()
                    X1 = p
            except EOFError:
                pass


print(X1)               
Dn = len(X1)
X1 =np.asarray(X1)
#Preprocessing
scaler = StandardScaler()
z = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,40000,700]
Temp_X1 = np.reshape(X1, (Dn* X1.shape[1], X1.shape[2]))
Temp_X1 = Temp_X1/z
X1 = np.reshape(Temp_X1, (Dn, X1.shape[1], X1.shape[2]))
#only use for one test data

X1 = np.reshape(X1, ((1,) + X1.shape))

sfc_feature = X1[0]
deployment = model.predict([sfc_feature])
deployment = np.argmax(deployment,axis=-1) #classifcation to regression

print(deployment)

