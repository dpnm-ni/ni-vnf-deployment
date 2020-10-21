import connexion
# import six

# from swagger_server import util
import op_vnf_deployment as vnf
from swagger_server.models.target_vnf_sfc_info import targetInfo # noqa: E501


def get_scenario():
    return vnf.gen_scenario()


def ml_vnf_deployment_pred(body):  # noqa: E501
    if connexion.request.is_json:
        body = targetInfo.from_dict(connexion.request.get_json())
        response = vnf.execute_vnf_deployment(body, "ML_pred")

    return response.tolist()


def ml_vnf_deployment(body):  # noqa: E501
    if connexion.request.is_json:
        body = targetInfo.from_dict(connexion.request.get_json())
        response = vnf.execute_vnf_deployment(body, "ML")

    return response.tolist()


def ilp_vnf_deployment(body):  # noqa: E501
    if connexion.request.is_json:
        body = targetInfo.from_dict(connexion.request.get_json())
        response = vnf.execute_vnf_deployment(body, "ILP")

    return response.tolist()


def get_topology():
    topology = vnf.get_topology()
    return topology.tolist()


def get_sfcr_list():
    sfcr_list = vnf.get_sfcr_list()
    return sfcr_list


def get_current_vnf_deployment(prefix):
    vnf.get_topology()
    deployment = vnf.get_current_vnf_deployment(prefix)
    return deployment.tolist()


def get_resource_usage(monitoring_time):
    vnf.get_topology()
    node_feature, edge_feature = vnf.get_resource_usage(monitoring_time)
    return [node_feature.tolist(), edge_feature.tolist()]


def predict_op_vnf_deployment():
    if connexion.request.is_json:
        topology = vnf.get_topology()
        node_feature, edge_feature = vnf.get_resource_usage()
        deployment = vnf.predict_op_vnf_deployment(node_feature, topology, edge_feature)

    return deployment.tolist()
