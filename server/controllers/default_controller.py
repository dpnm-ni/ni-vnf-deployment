import connexion
# import six

# from swagger_server import util
import deployment as deployment
import threading
from server.models.target_vnf_sfc_info import targetInfo # noqa: E501


def evaluate_deployment():
    
    response=deployment.auto_deployment(rating=True)
    
    return response


def auto_deployment():
    
    threading.Thread(target=deployment.auto_deployment, args=()).start()
    
    return "success"

    
def training_ml():
    
    response = deployment.training_ml()
    
    return response


def shutdown_deployment():

    if deployment.status_auto_deployment == False:
        print("Deployment is already not active")
    else :
        print("Shutdown deployment")
        deployment.status_auto_deployment = False
        
    return "success"
