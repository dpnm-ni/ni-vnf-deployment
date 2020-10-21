# ni-vnf-deployment-module
NI-VNF-Deployment-Module optimaly deploys VNF instances on the OpenStack testbed by using the pre-trained machine learning model. 

(This module is private and already configured to be used to DPNM testbed)

## Main Responsibilities
ML-based VNF deployment module.
- Provide APIs to deploy VNF deployment.
- Provide APIs to provide the input data to be fed into the deployment method.

## Requirements
```
Python 3.5.2+
```

Please install pip3 and requirements by using the command as below.
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

Please refer to the "Dependencies and Usage Instructions" for ILP Solver to use the function of /vnf_deployment/ILP.
```
https://github.com/dpnm-ni/testbed-ilp
```

## Configuration
This module runs as web server to handle multiple SFC requests during certain period time.
At this point (2020. 09), we assume the SFC requests are given as a text file, traf.txt.
To use a web UI of this module or send an SFC request to the module, a port number can be configured (a default port number is 8888)

```
# server/__main__.py

def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI VNF Sub-Module Service'})
    app.run(port=8888) ### Port number configuration
```

This module interacts with ni-mano to change VNF deployment in consideration of OpenStack environment.
To communicate with ni-mano, this module should know URI of ni-mano.
In ni-mano, ni_mon and ni_nfvo are responsible for interacting with this module so their URI should be configured as follows.

```
# config/config.yaml

ni_mon:
  host: http://<ni_mon_ip>:<ni_mon_port>      # Configure here to interact with a monitoring module
ni_nfvo:
  host: http://<ni_nfvo_ip>:<ni_nfvo_port>    # Configure here to interact with an NFVO module
```

Before running this module, OpenStack network ID should be configured because VNF instances in OpenStack can have multiple network interfaces.
At this point (2020. 09), we assume the VNF types and SFC types are given as follows, while planning to get the sfc types as input data.
That is because we need predefined VNF instance flavor and OS image/snapshot for each VNF type.

```
# op_vnf_deployment.py

# TODO -> to get this info as input parameters
vnf_types = ["fw_1", "fw_3", "ids_1", "ids_2", "nat_1", "nat_2", "nat_3"]

# TODO -> to get this info as input parameters
sfccat = {
            ("nat", "firewall", "ids"): 1,
            ("nat", "ids"): 2,
            ("nat", "firewall"): 3
        }

# flavor id/ image id definition for each VNF type
flavor_dict = {"fw_1": "3e5d7c22-19e1-4508-ae41-310c90ac5fd8"
                , "fw_3": "3e5d7c22-19e1-4508-ae41-310c90ac5fd8"
                , "ids_1": "b79058e0-67e5-4659-acb2-b354775eaf48"
                , "ids_2": "b79058e0-67e5-4659-acb2-b354775eaf48"
                , "nat_1": "c47424c4-e30a-44b1-a32c-ce3687889254"
                , "nat_2": "c47424c4-e30a-44b1-a32c-ce3687889254"
                , "nat_3": "c47424c4-e30a-44b1-a32c-ce3687889254"
             }
image_dict = {"fw_1": "4fecb43b-a940-412c-9c3a-b7aea50cdd74"
                , "fw_3": "4fecb43b-a940-412c-9c3a-b7aea50cdd74"
                , "ids_1": "59d3dbf6-f6fc-4cac-a9a2-9e460fb669f3"
                , "ids_2": "59d3dbf6-f6fc-4cac-a9a2-9e460fb669f3"
                , "nat_1": "033ef41e-b50b-4988-bdd3-cc3c300b701d"
                , "nat_2": "033ef41e-b50b-4988-bdd3-cc3c300b701d"
                , "nat_3": "033ef41e-b50b-4988-bdd3-cc3c300b701d"
             }

```

## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
python3 -m swagger_server
```

This module provides web UI based on Swagger:

```
http://<host IP running this module>:<port number>/ui/
```

To change VNF deployment in OpenStack testbed, this module processes a HTTP POST message including targetInfo data in its body.
You can change VNF deployment according to the deployment decision of machine learning model using web UI or using other library creating HTTP messages.
If you create and send a HTTP POST message to this module, the destination URI is as follows.

```
# Change the VNF deployment according to the deployment decision of machine learning model
http://<host IP running this module>:<port number>/vnf_deployment/machine_learning

# Return a topology as adjacency matrix
http://<host IP running this module>:<port number>/get_topology

# Return the sfc requests list
http://<host IP running this module>:<port number>/get_sfcr_list

# Return the resource usage as two arrays
http://<host IP running this module>:<port number>/get_resource_usage/{monitoring_time}

# Return the current vnf deployment as matrix
http://<host IP running this module>:<port number> /get_current_vnf_deployment/{prefix}
```

Required data to change VNF deployment is defined in targetInfo model that is JSON format data.
The targetInfo model consists of 4 data as follows.
Three kinds of prefix are for distinguishing target items to be considered as components of the concerned system. 

- **vnf_inst_prefix**: a prefix to identify VNF instances
- **sfcr_prefix**: a prefix to identify SFC requests, in this case, flow classifiers
- **sfc_prefix**: a prefix to identify SFC settings
- **mon_win_sec**: monitoring time period for resource usage data  

For example, a specific case of targetInfo data can be as follows.
```
  {
    "mon_win_sec": 10,
    "sfc_prefix": "sh-",
    "sfcr_prefix": "sh-",
    "vnf_inst_prefix": "sh-"
  }
```

## Remark

To change VNF deployment, we need to setup related SFCRs and SFCs again, and this module handles all SFCRs and SFCs with the prefix. 
The overall process is as belows. 
```
  destroy current SFCs -> destroy current SFCRs -> change VNF deployment ->  deploy new SFCRs -> deploy new SFCs
```
To complement the decision of machine learning model, a set of necessary VNF instances are recommended to be installed as "fixed".
If the name of certain VNF instance ends with "fixed", the module doesn't destroy the VNF instances.
When it comes to creating the SFCs, The module sets the SFC chain by grouping by the VNF type for load balancing.
A pretrained machine learning model is required as .h5 and json file as belows.
```
vnf_placement_model.h5
vnf_placement_model
```
When the /get_topology is called, topology txt file is created. The txt file can be used for training data generation. 
For now, output of get_sfcr_list() (op_vnf_deployment.py) is not being used.
