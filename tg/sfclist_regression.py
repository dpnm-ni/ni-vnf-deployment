import csv
import copy
import pickle
import numpy as np


Type1_name = ['nat1','firewall1','ids1']
Type2_name = ['nat2','ids2']
Type3_name = ['nat3','firewall3']

Type1 = [1,0,0]
Type2 = [0,1,0]
Type3 = [0,0,1]


Server0 = [1,0,0,0,0,0,0,0,0,0,0,0,0]
Server1 = [0,1,0,0,0,0,0,0,0,0,0,0,0]
Server2 = [0,0,1,0,0,0,0,0,0,0,0,0,0]
Server3 = [0,0,0,1,0,0,0,0,0,0,0,0,0]
Server4 = [0,0,0,0,1,0,0,0,0,0,0,0,0]
Server5 = [0,0,0,0,0,1,0,0,0,0,0,0,0]
Server6 = [0,0,0,0,0,0,1,0,0,0,0,0,0]
Server7 = [0,0,0,0,0,0,0,1,0,0,0,0,0]
Server8 = [0,0,0,0,0,0,0,0,1,0,0,0,0]
Server9 = [0,0,0,0,0,0,0,0,0,1,0,0,0]
Server10 =[0,0,0,0,0,0,0,0,0,0,1,0,0]
Server11 =[0,0,0,0,0,0,0,0,0,0,0,1,0]
Server12 =[0,0,0,0,0,0,0,0,0,0,0,0,1]


Base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#Base = [0,0,0,0,0,0,0,0,0]

def check_type(t):
    if np.array_equal(Type1_name,t):
        return Type1
    elif np.array_equal(Type2_name,t):
        return Type2
    elif np.array_equal(Type3_name,t):
        return Type3


def check_server(s):
    if s=='0':
        return Server0
    elif s=='1':
        return Server1
    elif s=='2':
        return Server2
    elif s=='3':
        return Server3
    elif s=='4':
        return Server4
    elif s=='5':
        return Server5
    elif s=='6':
        return Server6
    elif s=='7':
        return Server7
    elif s=='8':
        return Server8
    elif s=='9':
        return Server9
    elif s=='10':
        return Server10
    elif s=='11':
        return Server11
    elif s=='12':
        return Server12


f_name_f = '../../solver/middlebox-placement/src/logs/traf-wk1_sfccat_norm_inet2-'
f_name_1 = '0000'
f_name_2 = '000'
f_name_3 = '00'
f_name_4 = '0'
f_name_5 = ''
f_size = 3000 ##SIZE of DATA

f_name_n =''
full_f_name =''
pickle_request = []


#skip = [23189,30668]


load_file=open('nsol.txt','r')

f_size = load_file.read()

for i in range (1, int(f_size)+1):
    
    
    #if i in skip:
    #    print("skip : "+str(i))
    #    continue
    
    col=[]
    if i > 9999:
        f_name_n = f_name_5
    elif i > 999:
        f_name_n = f_name_4
    elif i > 99:
        f_name_n = f_name_3
    elif i > 9:
        f_name_n = f_name_2
    else:
        f_name_n = f_name_1

    full_f_name = f_name_f + f_name_n + str(i)      #simple : f_name_s // detail : f_name_d

    group_request = list(csv.reader(open(full_f_name,'r',encoding='UTF8')))

    temp = []
    group_max = 3

    if len(group_request) > group_max:
        print(len(group_request))

    group_length = min(len(group_request),group_max)

    
    for j in range(0, group_length):
        one_request = group_request[j]
        one_length = len(one_request)
        sfc_type = one_request[6:one_length]
        bandwidth = [int(one_request[3])]
        delay = [int(one_request[4])]
        src = one_request[1]
        dst = one_request[2]

        pickle_src = check_server(src)
        pickle_dst = check_server(dst)
        pickle_type = check_type(sfc_type)


        t1 = np.asarray(pickle_src)
        t2 = np.asarray(pickle_dst)
        test = np.sum([t1,t2],axis=0)
        test = test.tolist()

        finished_one_hot = pickle_type + pickle_src + pickle_dst + bandwidth + delay
        temp.append(finished_one_hot)

        #print(j)

    for j in range(group_length, group_max):
        temp.append(Base)

    #if (i % 8 ==0):
    pickle_request.append(temp)
    #print('completed on '+str(i)+'/'+str(f_size))

pickle_request = pickle_request[1:]
print('number of file_length'+str(len(pickle_request)))

with open('data/gnn_input/labeling_pickle.pickle', 'wb') as f:   
    pickle.dump(pickle_request, f, pickle.HIGHEST_PROTOCOL)
    

