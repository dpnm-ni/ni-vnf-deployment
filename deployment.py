import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from create_dashboard import create_dashboard
import datetime
import json
import time
import requests
import paramiko
import numpy as np
import datetime as dt
import math
import random
import pickle
import subprocess
import networkx as nx
import copy
import _thread
import matplotlib.pyplot as plt
import os
import glob
import functools
import multiprocessing
import itertools
import shutil
import mygnn
from pprint import pprint
from config import cfg
from multiprocessing.pool import ThreadPool

# OpenStack Parameters
openstack_network_id = cfg["openstack_network_id"] # Insert OpenStack Network ID to be used for creating SFC
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"
#ni_nfvo_client_api
ni_nfvo_client_cfg = ni_nfvo_client.Configuration()
ni_nfvo_client_cfg.host=cfg["ni_nfvo"]["host"]
ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))
ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(ni_nfvo_client_cfg))

#ni_monitoring_api
ni_mon_client_cfg = ni_mon_client.Configuration()
ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))


status_auto_deployment = False
sfc_update_flag = True
ILP_mydashboard_url = ""
ML_mydashboard_url = ""


def get_vnf_flavor(flavor_id):
    query = ni_mon_api.get_vnf_flavor(flavor_id)
    response = query

    return response


def deploy_vnf(vnf_spec):
    api_response = ni_nfvo_vnf_api.deploy_vnf(vnf_spec)

    return api_response


def destroy_vnf(id):
    api_response = ni_nfvo_vnf_api.destroy_vnf(id)

    return api_response


def get_nfvo_vnf_spec(flavor_name):
#    print("5")

    t = ni_nfvo_client.ApiClient(ni_nfvo_client_cfg)

    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(t)
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"][flavor_name]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


def set_vnf_spec(vnf_type, node_name, traffic_name):
    vnf_spec = get_nfvo_vnf_spec(vnf_type)
    vnf_spec.vnf_name = traffic_name+vnf_type
    vnf_spec.image_id = cfg["image"][vnf_type] #client or server
    vnf_spec.node_name = node_name

    return vnf_spec 

def check_available_resource(node_id):
#    print("20")

    node_info = get_node_info()
    selected_node = [ node for node in node_info if node.id == node_id ][-1]
    flavor = ni_mon_api.get_vnf_flavor(cfg["flavor"]["default"])

    if selected_node.n_cores_free >= flavor.n_cores and selected_node.ram_free_mb >= flavor.ram_mb:
        return True

    return False

# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
#    print("26")
    status = ni_mon_api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


def create_sfc(sfcr, instance_id_list):

    #Create SFCR using src,dst_id and 

    sfc_spec =ni_nfvo_client.SfcSpec(sfc_name=sfcr.name,
                                 sfcr_ids=[sfcr.id],
                                 vnf_instance_ids=instance_id_list,
                                 is_symmetric=False)


    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)

    print("Success to pass for creating sfc")
    return api_response


def get_ip_from_id(vm_id):
#    print("6")

    query = ni_mon_api.get_vnf_instance(vm_id)
    #print(query)

    ## Get ip address of specific network
    ports = query.ports
    #print(ports)
    network_id = openstack_network_id
    #print(network_id)

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


# get_sfc_by_name(sfc_name): get sfc information by using sfc_name from NFVO module
# Input: sfc name
# Output: sfc_info
def get_sfc_by_name(sfc_id):
#    print("11")

    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.id == sfc_id ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info



def get_sfcr_by_id(sfcr_id):
#    print("11")

    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.id == sfcr_id ]

    if len(sfcr_info) == 0:
        return False

    sfcr_info = sfcr_info[-1]

    return sfcr_info



def check_network_topology():

    api_response = ni_mon_api.get_links()   
    edges = {}
    dc = {}
    kreonet = {}
    for entry in api_response:
        node1_id = entry.node1_id
        node2_id = entry.node2_id
        
        if "kisti" in node1_id :
            kreonet.setdefault(node2_id,[]).append(node1_id) 
        elif "kisti" in node2_id :
            kreonet.setdefault(node1_id,[]).append(node2_id)  
        elif "core" in node1_id and "compute" in node2_id:
            dc.setdefault(node1_id,[]).append(node2_id)
        elif "core" in node2_id and "compute" in node1_id:
            dc.setdefault(node2_id,[]).append(node1_id)          
        elif "Switch" not in node2_id :
            edges.setdefault(node1_id,[]).append(node2_id)
        elif "Switch" not in node1_id :
            edges.setdefault(node2_id,[]).append(node1_id)

    return edges, dc, kreonet



def binary_encode_number(number, max_length=(len(ni_mon_api.get_nodes()).bit_length())):
    # Calculate the binary encoding for the number with padding
    binary_encoding = [int(bit) for bit in format(number, f'0{max_length}b')]
    return binary_encoding


def binary_encode_middlebox_selection(chain, middlebox = ['default','firewall','dpi','ids','proxy'], max_chain_length=3):
    #TODO
    #get middblex from cfg


    #get max_chain_length from cfg



    # 중복 순열 생성 (1개에서 3개까지)
    all_permutations = []
    for r in range(1, max_chain_length+1):
        permutations = list(itertools.permutations(middlebox, r))
        all_permutations.extend(permutations)
        
    bit_length = len(all_permutations).bit_length()

    # example_data가 몇 번째 순열과 일치하는지 찾음
    index = all_permutations.index(tuple(chain))

    # index를 이진(binary)으로 인코딩
    binary_encoded_data = list(map(int, bin(index)[2:].zfill(bit_length)))

    return binary_encoded_data


def generate_env_data():
    nodes = ni_mon_api.get_nodes()
    links = ni_mon_api.get_links()
    node_id_to_index = {node.id : index for index, node in enumerate(nodes)}
    #common-data-middle_box
    with open("env_data/middlebox-spec", 'w') as f:
        for flavor_name in cfg["flavor"]:
            #if flavor_name=="default":
            #    continue
            flavor_data = get_vnf_flavor(cfg["flavor"][flavor_name])
            f.write(flavor_name+","+str(flavor_data.n_cores)+","+str(max(1,int(flavor_data.delay_us)))+","+str(flavor_data.capacity_mbps)+",0.0\n")
    
    # inet8 파일 생성
    with open("env_data/testbed-clean", 'w') as f:
        f.write(str(len(nodes)) + " " + str(len(links)) + '\n')
        # nodes 정보 쓰기
        for index, node in enumerate(nodes):
            n_cores_free = node.n_cores if node.n_cores is not None else 0
            f.write(f"{index} {n_cores_free}\n")
        
        # links 정보 쓰기
        for link in links:
            f.write(f"{node_id_to_index[link.node1_id]} {node_id_to_index[link.node2_id]} {link.max_bw_mbps} {max(1,math.ceil(link.delay_us))}\n")
        
    return



def auto_deployment(rating=False):

    nodes = ni_mon_api.get_nodes()
    node_id_to_index = {node.id : index for index, node in enumerate(nodes)}
    global status_auto_deployment

    if status_auto_deployment == True:
        print("Auto_deployment is already active!")
        return
        
    nodes = ni_mon_api.get_nodes()
    generate_env_data()
    

    middleboxFile ='env_data/middlebox-spec'
    middlebox = []
    with open(middleboxFile, 'r') as f:
        for line in f:
            middlebox.append(line.strip().split(',')[0])    
    
    status_auto_deployment = True
    ll = 200
    while(status_auto_deployment):
        
        ilp_prediction = []
        ilp_result, ml_result = [], []
        ilp_sfc, ml_sfc = "", ""
        ml_vnf, ilp_vnf = [] , []
            
        
        sfcr = get_data_from_openstack()

        if sfcr == None:
            time.sleep(5)
            continue        
        
        sfcr_src_node_id = ni_mon_api.get_vnf_instance(sfcr.source_client).node_id
        sfcr_src_node_index = node_id_to_index[sfcr_src_node_id]
        

        #Already removed in get_data_from_openstack
        #sfcr.nf_chain.remove("src")
        #sfcr.nf_chain.remove("dst")

        topoFile = 'dataset/testbed/topology-0'
        validnodes = []
        with open(topoFile, 'r') as f:
        
            A = int(f.readline().strip().split()[0])
            
            for _ in range(A):
                line = f.readline().strip().split()
                second_value = int(line[1])
                if second_value > 0:
                    validnodes.append(int(line[0]))             

        if rating == True:
            status_auto_deployment = False  #<-
            runCPLEXPar(env="testbed")
            ilp_prediction = getLabelingFromDeployment(env="testbed")
            ilp_prediction = np.argmax(ilp_prediction, axis=-1)[0]
            indices = np.argwhere(ilp_prediction >= 1)
            instance_id_list = []
            vnf_type_list = []
            print("ILP VNF Deployment: ", ilp_prediction)


        #shutil.copytree("dataset/testbed", 'tt/'+str(ll))#<-
        #ll = ll + 1#<-

        #if ll < 300:#<-
        #    print(ll)#<-
        #    continue#<-

            for index in indices:
       
                #TODO selected_number ...
                selected_number = ilp_prediction[index[0]][index[1]]#1
                while(selected_number > 0):
                    selected_node = nodes[index[0]]
                    selected_vnftype = middlebox[index[1]]
                    vnf_type_list.append(selected_vnftype)
                    
                    traffic_name = ni_mon_api.get_vnf_instance(sfcr.source_client).name
                    traffic_name = traffic_name.replace("client","")
                    
                    spec = set_vnf_spec(selected_vnftype, selected_node.name, traffic_name)
                    print("Deployment Spec : ", spec)
                    instance_id = deploy_vnf(spec)
                    ilp_vnf.append(instance_id)
                    instance_id_list.append(instance_id) #When deployment cosider more than 2 vnf for same type, then this code should be modified.
                    limit = 500
                    for i in range(0, limit):
                        time.sleep(2)

                        # Success to create VNF instance
                        if check_active_instance(instance_id):
                            break
                        elif i == (limit-1):
                            destroy_vnf(instance_id)
                            print("Failed to deploy VNF")
                            return False
                    selected_number = selected_number - 1
                    
            indices = [vnf_type_list.index(val) if val in vnf_type_list else -1 for val in sfcr.nf_chain]  
            sorted_indices = np.argsort(indices)
            instance_id_list = [[instance_id_list[i]] for i in sorted_indices]
            
            #instance_id_list.insert(0,[sfcr.source_client])
            #instance_id_list.append([sfcr.destination_client])
            
            print("this is instance_id_list : ", instance_id_list)
            #print("sfcr : ", sfcr)
            
            ilp_sfc = create_sfc(sfcr, instance_id_list)
            
            grafana_list = [[ni_mon_api.get_vnf_instance(sfcr.source_client)],[],[],[],[ni_mon_api.get_vnf_instance(sfcr.destination_client)]]
            
            for iv in ilp_vnf:
                iv_info = ni_mon_api.get_vnf_instance(iv)
                iv_image = iv_info.image_id
                matching_keys = [key for key, value in cfg["image"].items() if value == iv_image][0]
                iv_index = middlebox.index(matching_keys)
                grafana_list[iv_index].append(iv_info)
                
            ILP_mydashboard_url = create_dashboard(grafana_list,"ILP_deployment")
      
            print("Finish set up vnf and sfc")
            
            ilp_result = measure_response_time(get_ip_from_id(sfcr.source_client), get_ip_from_id(sfcr.destination_client),"ILP_monitor.txt")
            
            #time.sleep(120)
            
            print("Deleting VNF and SFC")
            ni_nfvo_sfc_api.del_sfc(ilp_sfc)
            for iv in ilp_vnf:
                destroy_vnf(iv)


        time.sleep(10)
        
        prediction = mygnn.run_mygnn(env="testbed", is_trained=True, is_router_included=True) 
        prediction = prediction[0] 
        prediction = np.argmax(prediction, axis=-1)   
        print("ML VNF Deployment: ", prediction)   


        indices = np.argwhere(prediction >= 1)        
        
        instance_id_list = []
        vnf_type_list = []

        #Check for valid deployment action
        #if false then random deployment
        
        #check for nodes
        for i in range(indices.shape[0]):
            if indices[i, 0] not in validnodes:     
                prediction[indices[i, 0]][indices[i, 1]] = 0
                #print("Fail to find the valide nodes on indices : ",i)
                indices[i, 0] = sfcr_src_node_index#random.choice(validnodes)
                prediction[indices[i, 0]][indices[i, 1]] = 1


        #check for SFC
        if set(sfcr.nf_chain) != set([middlebox[index[1]] for index in indices]):
            indices = []
            #print("Fail to find the correct SFC type") 
            for chain in sfcr.nf_chain:
                indices.append([sfcr_src_node_index, middlebox.index(chain)])
                prediction[sfcr_src_node_index][middlebox.index(chain)] = 1
                
        #print("AFTER RESULT : ", prediction) 

        if rating == True: ####Should removed
            prediction = ilp_prediction  
            indices = np.argwhere(ilp_prediction >= 1) 

        for index in indices:
   
            #TODO selected_number ...
            selected_number = prediction[index[0]][index[1]]#1
            while(selected_number > 0):
                selected_node = nodes[index[0]]
                selected_vnftype = middlebox[index[1]]
                vnf_type_list.append(selected_vnftype)
                
                traffic_name = ni_mon_api.get_vnf_instance(sfcr.source_client).name
                traffic_name = traffic_name.replace("client","")
                
                spec = set_vnf_spec(selected_vnftype, selected_node.name, traffic_name)
                instance_id = deploy_vnf(spec)
                ml_vnf.append(instance_id)
                instance_id_list.append(instance_id) #When deployment cosider more than 2 vnf for same type, then this code should be modified.
                limit = 500
                for i in range(0, limit):
                    time.sleep(2)

                    # Success to create VNF instance
                    if check_active_instance(instance_id):
                        break
                    elif i == (limit-1):
                        destroy_vnf(instance_id)
                        print("Failed to deploy VNF")
                        return False
                selected_number = selected_number - 1
                
        indices = [vnf_type_list.index(val) if val in vnf_type_list else -1 for val in sfcr.nf_chain]  
        sorted_indices = np.argsort(indices)
        instance_id_list = [[instance_id_list[i]] for i in sorted_indices]
        
        #instance_id_list.insert(0,[sfcr.source_client])
        #instance_id_list.append([sfcr.destination_client])
        
        #print("this is instance_id_list : ", instance_id_list)
        #print("sfcr : ", sfcr)
        
        ml_sfc = create_sfc(sfcr, instance_id_list)
        
        if rating == True:
            ml_result = measure_response_time(get_ip_from_id(sfcr.source_client), get_ip_from_id(sfcr.destination_client),"ML_monitor.txt")

            grafana_list = [[ni_mon_api.get_vnf_instance(sfcr.source_client)],[],[],[],[ni_mon_api.get_vnf_instance(sfcr.destination_client)]]
            
            for iv in ml_vnf:
                iv_info = ni_mon_api.get_vnf_instance(iv)
                iv_image = iv_info.image_id
                matching_keys = [key for key, value in cfg["image"].items() if value == iv_image][0]
                iv_index = middlebox.index(matching_keys)
                grafana_list[iv_index].append(iv_info)
                
            ML_mydashboard_url = create_dashboard(grafana_list,"ML_deployment")

            print("Deleting VNF and SFC")
            ni_nfvo_sfc_api.del_sfc(ml_sfc)
            for iv in ml_vnf:
                destroy_vnf(iv)       
        
        

        time.sleep(5)


    return ("ILP average dealy : {}, ML average delay : {}, Accuracy : {} ILP grafana dashboard : {} ML grafana dashboard : {}".format(round(float(ilp_result),3), round(float(ml_result),3), round(float(min(ilp_result/ml_result,ml_result/ilp_result)),3), ILP_mydashboard_url, ML_mydashboard_url))


def training_ml():

    generate_env_data()
    get_data_from_simulation(num_data = 100000)#99997
    runCPLEXPar(env="simulation", ncores=64)
    getLabelingFromDeployment()

    result = mygnn.run_mygnn(is_trained=False, is_router_included=True)
    
    return "Accuracy : {}".format(result)


def get_data_from_openstack():
    folder_path = 'dataset/testbed'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)        
    os.mkdir(folder_path)

    nodes = ni_mon_api.get_nodes()
    links = ni_mon_api.get_links()
    sfcrs = ni_nfvo_sfcr_api.get_sfcrs()
    sfcs = ni_nfvo_sfc_api.get_sfcs()
    
    node_matrix = np.zeros((len(nodes), 1), dtype=int)#[node.n_cores_free for nodes]
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    edge_matrix = np.zeros((len(nodes), len(nodes), 2), dtype=int)

    # 노드 ID를 인덱스로 매핑하는 딕셔너리 생성
    node_id_to_index = {node.id : index for index, node in enumerate(nodes)}

    edges, dc, kreonet = check_network_topology()  
    edge_and_kreonet = {**edges, **kreonet}
    candidate_nodes = [value for values in edge_and_kreonet.values() for value in values]


    # Get real free bandwidth links  
    src, src_client, dst, matched_sfcr, ex_sfcs, chain, bandwidth, traffic_data = None, None, None, None, [], None, 0, [0]


    candidated_sfcr_id = [sfcr.id for sfcr in sfcrs]
    
    for sfc in sfcs:
        ex_sfcs.append(sfc)
        candidated_sfcr_id.remove(sfc.sfcr_ids[0])
        
    #If there are non-used sfcr then has problem. so it should be fixed.
    if len(candidated_sfcr_id) > 0:
        matched_sfcr = get_sfcr_by_id(candidated_sfcr_id[0])
        src_client = matched_sfcr.source_client
        src = ni_mon_api.get_vnf_instance(matched_sfcr.source_client).node_id
        dst = ni_mon_api.get_vnf_instance(matched_sfcr.destination_client).node_id
        chain = matched_sfcr.nf_chain
        #chain.remove("src")
        #chain.remove("dst")
    else :
        print("[Auto Deployment Running] Cannot find new service requests.")
        return None
 

    if ex_sfcs:
        G = nx.Graph()
        for node_data in nodes:
            G.add_node(node_data.id, n_cores_free=node_data.n_cores_free)
        for link_data in links:
            G.add_edge(link_data.node1_id, link_data.node2_id, delay_us = link_data.delay_us, max_bw_mbps = link_data.max_bw_mbps)
        for ex_sfc in ex_sfcs:
            ex_vnfs = ex_sfc.vnf_instance_ids
            ex_src_client = ex_vnfs[0][0]
        
            start_time = datetime.datetime.now() #+ datetime.timedelta(hours=24)
            end_time = start_time +datetime.timedelta(seconds=1)
            time.sleep(2)

            if str(start_time)[-1]!='Z':
                start_time = str(start_time.isoformat())+'Z'
                end_time = str(end_time.isoformat())+'Z'
                
            measurement_types = ni_mon_api.get_measurement_types(id=ex_src_client)
            measurement_txs = [item for item in measurement_types if "if_octets___tx___derive" in item]   
            
            ex_bandwidth = 0
            try:    
                for measurement_tx in measurement_txs:
                    query = ni_mon_api.get_measurement(id=ex_src_client, measurement_type=measurement_tx, start_time=start_time, end_time=end_time)
                    ex_bandwidth = ex_bandwidth + query[0].measurement_value
                ex_bandwidth = int(ex_bandwidth*8/1000000)
                print("ex_bandwidth : ", ex_bandwidth)
            except:
                print("Skipping the dead VNF {}".format(ex_src_client))
                continue
            
            #find the paths
            ex_path = []
            updated_edges = set()

            for ex_vnf_index in range(len(ex_vnfs)-1):
                source_node_id = ni_mon_api.get_vnf_instance(ex_vnfs[ex_vnf_index][0]).node_id
                destination_node_id = ni_mon_api.get_vnf_instance(ex_vnfs[ex_vnf_index+1][0]).node_id
                
                partial_path = list(nx.all_shortest_paths(G, source= source_node_id, target= destination_node_id))[0]
                
                if ex_path:
                    ex_path = ex_path + partial_path[1:]
                else:
                    ex_path = partial_path
                
            for current_index in range(len(ex_path) - 1):
                edge = (ex_path[current_index], ex_path[current_index + 1])
                if edge not in updated_edges:
                    G[ex_path[current_index]][ex_path[current_index + 1]]['max_bw_mbps'] -= ex_bandwidth
                    updated_edges.add(edge)
                    
            for link in links:
                link.max_bw_mbps = G[link.node1_id][link.node2_id]['max_bw_mbps']
                 

    for node_id in node_id_to_index:
        index = node_id_to_index[node_id]
        n_cores_free = nodes[index].n_cores_free
        node_matrix[index, 0] = n_cores_free if n_cores_free is not None else 0

    # 링크 정보 반영
    for link in links:
        node1_index = node_id_to_index[link.node1_id]
        node2_index = node_id_to_index[link.node2_id]
        adjacency_matrix[node1_index, node2_index] = 1
        adjacency_matrix[node2_index, node1_index] = 1
        edge_matrix[node1_index, node2_index] = (link.max_bw_mbps,link.delay_us)#(link.max_bw_mbps, max(1,math.ceil(link.delay_us/1000)))
        edge_matrix[node2_index, node1_index] = (link.max_bw_mbps,link.delay_us)#(link.max_bw_mbps, max(1,math.ceil(link.delay_us/1000)))

    with open("dataset/testbed/node_matrix", 'wb') as f:
        pickle.dump([node_matrix], f)
    with open("dataset/testbed/adjacency_matrix", 'wb') as f:
        pickle.dump([adjacency_matrix], f) 
    with open("dataset/testbed/edge_matrix", 'wb') as f:
        pickle.dump([edge_matrix], f)  


    measurement_types = ni_mon_api.get_measurement_types(id=src_client)
    measurement_txs = [item for item in measurement_types if "if_octets___tx___derive" in item]    

    count = 0
    while(bandwidth < 10): #meaningful octect should bigger than 1000  
        bandwidth = 0
        start_time = datetime.datetime.now() #+ datetime.timedelta(hours=24)
        end_time = start_time +datetime.timedelta(seconds=2) 
        
        print("waitting for iperf3/traffic set up from target sfc")
        time.sleep(3)

        if str(start_time)[-1]!='Z':
            start_time = str(start_time.isoformat())+'Z'#"2023-08-30T13:51:07.668941Z"#
            end_time = str(end_time.isoformat())+'Z'#"2023-08-30T13:51:08.668941Z"#
            
        try :
            for measurement_tx in measurement_txs:
                query = ni_mon_api.get_measurement(id=src_client, measurement_type=measurement_tx, start_time=start_time, end_time=end_time)
                bandwidth = bandwidth + query[0].measurement_value
            bandwidth = int(bandwidth*8/1000000)
            count += 1
        except :
            print("API or ni-collcter has problem!!")
            return
        
        if count >1:
            print("Cannot find the active traffic now. left count : {}".format(11-count))

           
    
    traffic_data.append(node_id_to_index[src])
    traffic_data.append(node_id_to_index[dst])
    traffic_data.append(bandwidth)
    traffic_data.append(30000)
    traffic_data.append(0.00000010)
    traffic_data.extend(chain)

    
    traffics = []
    traffics.extend(binary_encode_number(node_id_to_index[src]))
    traffics.extend(binary_encode_number(node_id_to_index[dst]))
    traffics.append(bandwidth)
    traffics.extend(binary_encode_middlebox_selection(chain))   
    
    with open("dataset/testbed/traffic-0", 'w') as f:
        f.write(','.join(map(str,traffic_data)))

    with open("dataset/testbed/traffics", 'wb') as f:
        pickle.dump(np.array([traffics]), f)  


    #src_index, dst_index = random.sample(range(len(candidate_nodes)), 2)
    #src = candidate_nodes[src_index]
    #dst = candidate_nodes[dst_index]
     
    excluded_nodes = [value for key, values in edge_and_kreonet.items() if src not in values and dst not in values for value in values]

    # inet8 파일 생성
    with open("dataset/testbed/topology-0", 'w') as f:
        f.write(str(len(nodes)) + " " + str(len(links)) + '\n')
        # nodes 정보 쓰기
        for index, node in enumerate(nodes):
            n_cores_free = node.n_cores_free if node.n_cores_free is not None and node.id not in excluded_nodes else 0
            f.write(f"{index} {n_cores_free}\n")
        
        # links 정보 쓰기
        for link in links:
            f.write(f"{node_id_to_index[link.node1_id]} {node_id_to_index[link.node2_id]} {link.max_bw_mbps} {link.delay_us}\n")


    return matched_sfcr

def get_data_from_simulation(num_data = 1):

    folder_path = 'dataset/simulation'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        
    os.mkdir(folder_path)


    #Get Structure data from Openstack
    nodes = ni_mon_api.get_nodes()
    links = ni_mon_api.get_links()
    
    G = nx.Graph()
    
    
    sfcrs = ni_nfvo_sfcr_api.get_sfcrs()
    sfcs = ni_nfvo_sfc_api.get_sfcs()
    
    node_matrix = []
    adjacency_matrix = []
    edge_matrix = []
    traffics = []
    max_num_traffic = 0
    
    edges, dc, kreonet = check_network_topology()  
    edge_and_kreonet = {**edges, **kreonet}
    #candidate_nodes = [value for values in edge_and_kreonet.values() for value in values]
    candidate_nodes = [value for values in edges.values() for value in values]
    all_nodes = [value for values in {**edges, **kreonet, **dc}.values() for value in values]

    # 노드 ID를 인덱스로 매핑하는 딕셔너리 생성
    node_id_to_index = {node.id : index for index, node in enumerate(nodes)}
    
    middlebox_info = {}
    with open("env_data/middlebox-spec", 'r') as f:
        for line in f:
            middlebox, cpu, delay_us, capacity, cost = line.strip().split(',')  # 쉼표로 구분된 값을 리스트로 분리합니다.
            cpu = int(cpu)
            delay_us = int(delay_us)
            capacity = int(capacity)
            cost = float(cost)

            middlebox_info[middlebox] = {
                "cpu": cpu,
                "delay_us": delay_us,
                "capacity": capacity,
                "cost": cost
            }


    #Set Simulation parameter
    for i in range(num_data):
        src_index, dst_index = random.sample(range(len(candidate_nodes)), 2)
        src = candidate_nodes[src_index]
        dst = candidate_nodes[dst_index]
        excluded_nodes = [value for key, values in edge_and_kreonet.items() if src not in values and dst not in values for value in values]
        
        bandwidth = 100#100 or random.randint(100,300)
        chain = random.choices(cfg["sfc"]["types"], cfg["sfc"]["prob"])[0]
        
        traffic_data = [0]
        traffic_data.append(src_index)
        traffic_data.append(dst_index)
        traffic_data.append(bandwidth)#b
        traffic_data.append(36000)
        traffic_data.append(0.00000010)
        traffic_data.extend(chain)
        
        traffic_feature = []
        traffic_feature.extend(binary_encode_number(src_index))
        traffic_feature.extend(binary_encode_number(dst_index))
        traffic_feature.append(bandwidth)#b
        traffic_feature.extend(binary_encode_middlebox_selection(chain))        
        
        
        with open("dataset/simulation/traffic-{}".format(i), 'w') as f:
            f.write(','.join(map(str,traffic_data)))
             
        required_cpu = [middlebox_info[vnf_type]['cpu'] for vnf_type in chain]
        required_cpu = sum(required_cpu)


        #link apply
        while True:
            G.clear()
            for node in nodes:
                G.add_node(node.id, n_cores_free = random.randint(node.n_cores-2, node.n_cores) if node.n_cores_free is not None else 0 )#(8, 32 OK)
            
            #print("TEST : ", nx.get_node_attributes(G,'n_cores_free').values())
            if not sum((nx.get_node_attributes(G,'n_cores_free')).values()) < required_cpu:
                break

        # 엣지 추가
        for link_data in links:
            G.add_edge(link_data.node1_id, link_data.node2_id, delay_us = link_data.delay_us, max_bw_mbps = link_data.max_bw_mbps)
                      
        num_ex_traffic = random.randint(0,max_num_traffic)
        
        for ex_traffic in range(0, num_ex_traffic):
            ex_src_index, ex_dst_index = random.sample(range(len(candidate_nodes)), 2)
            ex_src = candidate_nodes[ex_src_index]
            ex_dst = candidate_nodes[ex_dst_index]
            ex_excluded_nodes = [value for key, values in edge_and_kreonet.items() if ex_src not in values and ex_dst not in values for value in values]
            ex_inter = random.choice([x for x in all_nodes if x not in ex_excluded_nodes])
            ex_bandwidth = random.randint(20,50)
            
            ##For Kisti
            if "kisti" in ex_src or "kisti" in ex_dst:
                ex_bandwidth = random.randint(10, 25)
            
            partial_path1 = list(nx.all_shortest_paths(G, source=ex_src, target=ex_inter))[0]
            partial_path2 = list(nx.all_shortest_paths(G, source=ex_inter, target=ex_dst))[0]

            
            if len(partial_path1) > 2:
                partial_path1 = partial_path1[:-1]
            elif len(partial_path2) > 2:
                partial_path2 = partial_path2[1:]
            
            path = partial_path1 + partial_path2
            updated_edges = set()
            
            for current_index in range(len(path) - 1):
                edge = (path[current_index], path[current_index + 1])
                if edge not in updated_edges:

                    G[path[current_index]][path[current_index + 1]]['max_bw_mbps'] -= ex_bandwidth
                    updated_edges.add(edge)
                    

        k = len(nodes)
        
        single_node_matrix = np.array([[nx.get_node_attributes(G, 'n_cores_free')[node.id]] for node in nodes], dtype=int)
        #single_node_matrix = np.array([[G[node.id]['n.cores_free']] for node in nodes], dtype=int)
        single_adjacency_matrix = np.zeros((k, k), dtype=int)
        single_edge_matrix = np.zeros((k, k, 2), dtype=int)

        for link in links:
            node1_index = node_id_to_index[link.node1_id]
            node2_index = node_id_to_index[link.node2_id]
            single_adjacency_matrix[node1_index, node2_index] = 1
            single_adjacency_matrix[node2_index, node1_index] = 1
            single_edge_matrix[node1_index, node2_index] = (G[link.node1_id][link.node2_id]['max_bw_mbps'],G[link.node1_id][link.node2_id]['delay_us'])
            single_edge_matrix[node2_index, node1_index] = (G[link.node2_id][link.node1_id]['max_bw_mbps'],G[link.node2_id][link.node1_id]['delay_us'])    #(link.max_bw_mbps,link.delay_us)


        node_matrix.append(single_node_matrix)
        adjacency_matrix.append(single_adjacency_matrix)
        edge_matrix.append(single_edge_matrix)
        traffics.append(traffic_feature)
        
  
           # inet8 파일 생성
        with open("dataset/simulation/topology-{}".format(i), 'w') as f:
            f.write(str(len(nodes)) + " " + str(len(links)) + '\n')
            # nodes 정보 쓰기
            for index, node in enumerate(nodes): #enumerate(G.nodes)
                #n_cores_free = nx.get_node_attributes(G, 'n_cores_free')[node]
                n_cores_free = node.n_cores_free if node.n_cores_free is not None and node.id not in excluded_nodes else 0
                f.write(f"{index} {n_cores_free}\n")
            
            # links 정보 쓰기
            for link in links:
                f.write(f"{node_id_to_index[link.node1_id]} {node_id_to_index[link.node2_id]} {G[link.node1_id][link.node2_id]['max_bw_mbps']} {G[link.node1_id][link.node2_id]['delay_us']}\n")


    with open("dataset/simulation/node_matrix", 'wb') as f:
        pickle.dump(np.array(node_matrix), f)
    with open("dataset/simulation/adjacency_matrix", 'wb') as f:
        pickle.dump(np.array(adjacency_matrix), f) 
    with open("dataset/simulation/edge_matrix", 'wb') as f:
        pickle.dump(np.array(edge_matrix), f)  
    with open("dataset/simulation/traffics", 'wb') as f:
        pickle.dump(np.array(traffics), f)    
        
    return



def runCPLEXPar(env="simulation", ncores=8):
    folder_path = 'dataset/' + env
    
    trafficPath = folder_path+'/traffic-*'
    topoPath = folder_path+'/topology-*'
    logPath = folder_path+'/log-*'
    middleboxFile = 'env_data/middlebox-spec'
   
    trafficFiles = glob.glob(trafficPath)
    logFiles = glob.glob(logPath)
    
    completedIDs = [int(x.split('-')[-1].split('.')[0]) for x in logFiles]
    

    remainingIndices = list(set(range(0,len(trafficFiles))) - set(completedIDs))
    
    cplexSingle = functools.partial(runCPLEXSingleInstance, trafficPath.replace('*', ''), topoPath.replace('*', ''), logPath.replace('*', ''), middleboxFile)
    pool = multiprocessing.Pool(processes = ncores)
    res = pool.map(cplexSingle, remainingIndices) #remainingIndices is used for cplex input i as identifier 


# Just like runCPLEX, but performs only the i-th iteration.
def runCPLEXSingleInstance(trafficPath, topoPath, logPath, middleboxFile, i):
    trafficLists = glob.glob(trafficPath)
    trafficFile = '../'+ trafficPath+str(i)
    topoFile = '../'+ topoPath+str(i)
    logFile = logPath+str(i)
    middleboxFile = '../'+ middleboxFile
    
    nFiles = len(trafficLists)
    print('%07d/%07d' % (i, nFiles))
    outfile = open(logFile, 'w')
    runcommand = './middleman --per_core_cost=0.01 --per_bit_transit_cost=3.626543209876543e-7 --topology_file=%s --middlebox_spec_file=%s --traffic_request_file=%s --max_time=60 --algorithm=cplex' % (topoFile, middleboxFile, trafficFile)
    subprocess.call(runcommand.split(),cwd='solver_src', stdout=outfile)
    getPlacementInfoFromCPLEXLogs(logFile,i)
    
    
    return
    
    
def getPlacementInfoFromCPLEXLogs(logFile,i):
    deployment_path = logFile.replace("log", "deployment") 
    with open(logFile) as f:
        with open(deployment_path,"w") as f2:
            # Go through logfile once, extracting relevant information.
            vnflocs = []
            for line in f.readlines():
                if 'vnfloc' in line:
                    vnflocs.append(line)

            if len(vnflocs) == 0:
                print('[ WARN ] no vnfloc tag in file %s - skipping.' % i)
                return
                
            # Last line gives final number of deployed middleboxes.
            nMiddleBoxes = vnflocs[-1]
            nMiddleBoxes = int(nMiddleBoxes.split(' ')[3])
            # Given the number of middleboxes, extract their location and type.
            middleboxLocations = []
            for line in vnflocs[-(nMiddleBoxes + 1):-1]:
                (location, vnfType) = (int(line.split()[5]), line.split()[3][:-1])
                middleboxLocations.append((location, vnfType))

            # Number of instaces of a VNF per node.
            nodeinfo = [(item[0], item[1], middleboxLocations.count(item)) for item in set(middleboxLocations)]
            for line in nodeinfo:
                f2.write('%d,%s,%d\r\n' % (line[0], line[1], line[2]))


    os.remove(logFile)

    return


def getLabelingFromDeployment(env="simulation"):

    topologyFile ='env_data/testbed'
    num_node = 0
    with open(topologyFile, 'r') as f:
        num_node =f.readlines()[0].strip().split(' ')[0]

    
    middleboxFile ='env_data/middlebox-spec'
    middlebox = []
    with open(middleboxFile, 'r') as f:
        for line in f:
            middlebox.append(line.strip().split(',')[0])
            
            
    deploymentPath = 'dataset/'+env+'/deployment-*'
    deploymentFiles = glob.glob(deploymentPath)
    print("files : ", deploymentFiles)
    
    num_of_nodes = int(num_node)
    num_of_types = len(middlebox)
    max_vnfs_deployed = 3
    deployment_logs = []

    for i in range(0,len(deploymentFiles)):
        deployment_log = [[[1 if m ==0 else 0 for m in range(max_vnfs_deployed)] for _ in range(num_of_types)] for _ in range(num_of_nodes)]

        with open("dataset/"+env+"/deployment-"+str(i), 'r') as f:
            for line in f:
                node, vnf, number = line.strip().split(',')
                #print("i : {} node : {}, vnf : {}, number : {}".format(i, node, vnf, number))
                deployment_log[int(node)-1][middlebox.index(vnf)][int(number)] = 1
                deployment_log[int(node)-1][middlebox.index(vnf)][0] = 0
                 
        deployment_logs.append(deployment_log)
    with open("dataset/"+env+"/deployments", 'wb') as f:
        pickle.dump(np.array(deployment_logs), f)    
                
    return deployment_logs
    
    
    
def measure_response_time(src_ip, dst_ip,file_name="test_monitor.txt"):

    count = 0
    total_delay = 0.0
    command = ("sshpass -p %s ssh -o stricthostkeychecking=no %s@%s ./test_http_e2e.sh %s %s %s %s %s" % (cfg["traffic_controller"]["password"],
                                                                            cfg["traffic_controller"]["username"],
                                                                            cfg["traffic_controller"]["ip"],
                                                                            src_ip,
                                                                            cfg["instance"]["username"],
                                                                            cfg["instance"]["password"],
                                                                            cfg["traffic_controller"]["num_requests"],
                                                                            dst_ip))

    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"

    print(command)
    # Wait until web server is running
    start_time = dt.datetime.now() 
    coount = 0
    while True:
        
        
        time.sleep(1)
        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")
        
        try:
            total_delay = total_delay + float(response)
            if count == 0 :
                start_time = dt.datetime.now()
            count = count + 1
        except:
            total_delay = total_delay

        if response != "":
            if count > 1: #For skipping inital measurement
                pprint("[{}] : {}".format(file_name, response))
                f = open(file_name, "a+", encoding='utf-8')
                f.write(str(response)+'\n')
                coount = coount + 1
                f.close()
            
            if coount == 6:
                return float(total_delay/count)

   
    
    
