import base64
import os
from hashlib import sha256
import requests
from requests_toolbelt import exceptions
from requests_toolbelt.downloadutils import stream
import itertools
class GNN_Methods:
    Graph_SAINT = "G_SAINT"
    RGCN = "RGCN"
    ShaDowGNN = "ShaDowSAINT"
    SeHGGN = "SeHGNN"
    IBS = "IBS"
    MorsE = "MorsE"
    NodePiece = "NodePiece"
    LHGNN = "LHGNN"
    WISE_SHSAINT = "WISE_SHSAINT"
    def __init__(self):
        ""
class KGE_Methods:
    TransE = "TransE"
    CompEx = "CompEx"
    RotatE = "RotatE"
    DistMult = "DistMult"
    def __init__(self):
        ""
class GML_Operator_Types:
    NodeClassification = "NodeClassifier"
    LinkPrediction = "LinkPredictor"
    GraphClassification = "GC"
    def __init__(self):
        ""
class GNN_Samplers:
    BRW = "BRW"
    RW = "RW"
    WRW = "WRW"
    PPR = "PPR"
    def __init__(self):
       ""

class GNN_KG_HParms:
    GNN_KG_PREFIX = 'kgnet:GMLModel/'
    GNN_Method = GNN_KG_PREFIX + 'GNNMethod'
    GNN_KG_HP_PREFIX = GNN_KG_PREFIX + 'hyperparameter/'
    Emb_size = GNN_KG_HP_PREFIX + 'embSize'
    HiddenChannels = GNN_KG_HP_PREFIX + 'hiddenChannels'
    Num_Classes = ''
    Num_Layers = GNN_KG_HP_PREFIX + 'numLayers'
    Dropout = ''

    def __init__(self):
       ""
class GNN_SubG_Parms:
    GNN_TYPE_PREFIX = 'kgnet:type/'
    GNN_KG_HP_PREFIX = GNN_KG_HParms.GNN_KG_PREFIX + 'taskSubgraph/'
    GNN_TASK_PREFIX = 'kgnet:GMLTask/'
    GNN_KG_TASK_PREFIX = 'kgnet:GMLTask/'
    targetEdge = GNN_KG_HP_PREFIX + 'targetEdge'
    targetNode = GNN_TASK_PREFIX + 'targetNode'
    prefix = GNN_KG_TASK_PREFIX + 'KGPrefix'
    modelId = GNN_TASK_PREFIX + 'modelID'
    taskType = GNN_TASK_PREFIX + 'taskType'
    labelNode = GNN_TASK_PREFIX + 'labelNode'
    def __init__(self):
       ""
class GML_Query_Types:
    Inference = "Select"
    Insert = "Insert"
    Delete = "Delete"
    def __init__(self):
       ""
class FileStorageType:
    localfile="localfile"
    remoteFileStore="remoteFileStore"
    S3="S3"
class TOSG_Patterns:
    d1h1 = "d1h1"
    d1h2 = "d1h2"
    d2h1 = "d2h1"
    d2h2 = "d2h2"
    def __init__(self):
       ""
class KGNET_Config:
    # datasets_output_path = "/mnt/KGNET/Datasets/"
    # inference_path = datasets_output_path + 'Inference/'
    # trained_model_path = datasets_output_path + 'trained_models/'
    # GML_API_URL = "http://206.12.102.12:64647/"
    # GML_Inference_PORT = "64647"
    # GML_ModelManager_PORT = "64648"
    # # GML_ModelManager_URL = "http://206.12.100.114"
    # GML_ModelManager_URL = "http://0.0.0.0"
    # KGMeta_IRI = "http://kgnet/"
    # KGMeta_endpoint_url = "http://206.12.98.118:8890/sparql/"

    # datasets_output_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Datasets')#"/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/"
    # datasets_output_path = r'/shared_mnt/github_repos/KGNET/Datasets/'
    datasets_output_path = r'/media/hussein/74c609e2-11c7-4345-9782-e0dffa447088/Github_Rpos/KGNET/Datasets/'
    inference_path = datasets_output_path + 'Inference/'
    # inference_path = os.path.join(datasets_output_path,'Inference')
    trained_model_path = datasets_output_path + 'trained_models/'
    # trained_model_path = os.path.join(datasets_output_path,'trained_models')
    emb_store_path = os.path.join(trained_model_path,'emb_store')
    GML_API_URL = "http://206.12.102.12:64648/"
    # GML_API_URL = "http://localhost:64648/"
    GML_Inference_PORT = "64648"
    GML_ModelManager_PORT = "8443"
    #GML_ModelManager_PORT = "8443"
    # GML_ModelManager_URL = "http://206.12.100.114"
    # GML_ModelManager_URL = "http://206.12.102.12"
    GML_ModelManager_URL = "http://206.12.102.12"
    KGMeta_IRI = "http://kgnet/"
    KGMeta_endpoint_url = "http://206.12.98.118:8890/sparql/"
    fileStorageType=FileStorageType.localfile#FileStorageType.remoteFileStore
    def __init__(self):
       ""


KGs_prefixs_dic={"dblp":"https://dblp.org/rdf/schema#",
             "lkmdb":"http://data.linkedmdb.org/resource/movie/",
             "mag":"https://www.mag.org/",
             "aifb":"http://www.aifb.uni-karlsruhe.de/",
             "yago":"http://schema.org/",
             "crunchbase":"http://ontologycentral.com/2010/05/cb/vocab#",
             "biokg":"http://www.biokg.com/"}
namedGraphURI_dic={"dblp":"http://dblp.org/",
             "lkmdb":"https://linkedmdb.org",
             "mag":"https://www.mag.org/",
             "aifb":"http://www.aifb.uni-karlsruhe.de/",
             "yago":"https://yago-knowledge.org",
             "crunchbase":"http://crunchbase-dump-2015-10",
             "biokg":"http://www.biokg.com"}


class colors:
    green="#A3EBB1"
    red="#FF5C5C"
    orange="#ffbd2c"
    def __init__(self):
       ""
class aggregations:
    min='min'
    max='max'
    avg='avg'
    mean='mean'
    def __init__(self):
     ""
class RDFEngine:
    OpenlinkVirtuoso="OpenlinkVirtuoso"
    stardog = "stardog"
class utils:
    def __init__(self):
       ""
    @staticmethod
    def highlight_value_in_column(column, color='red', agg='max'):
        highlight = 'background-color: ' + color + ';'
        default = ''
        if agg == 'max':
            val_in_column = column.max()
        elif agg == 'min':
            val_in_column = column.min()
        if agg == 'avg':
            val_in_column = column.avg()
        # must return one string per cell in this column
        return [highlight if v == val_in_column else default for v in column]

    @staticmethod
    def highlightRowByIdx(row, idx, bgcolor='palegreen', textcolor='red', fontweight='bold'):
        highlight = ['background-color:' + bgcolor + ' ;  font-weight:' + fontweight + '; color: ' + textcolor + ';']
        default = ['']
        # print(row.name)
        if row.name == idx:
            return highlight * len(row)
        else:
            return default * len(row)
    # df.style.apply(lambda row: highlightRowByIdx(row, 0), axis=1)

    @staticmethod
    def highlightDiferrentRowValues(row, bgcolor='palegreen', textcolor='red', fontweight='bold'):
        highlight = 'background-color:' + bgcolor + ' ;  font-weight:' + fontweight + '; color: ' + textcolor + ';'
        default = ''
        # must return one string per cell in this row
        if row[0] != row[1]:
            if len(str(row[0]).strip().replace("\"",""))==0:
                return [default, 'background-color:' + bgcolor + ' ;  font-weight:' + fontweight + ';']
            else:
                return [default, 'background-color:red ;  font-weight:' + fontweight + ';']
        elif str(row[0]) == str(row[1]):
            return ['background-color:#EDF9EB;', 'background-color:#EDF9EB;']
        else:
            return [default, default]
    # df.style.apply(highlightDiferrentRowValues, subset=['num_children', 'num_pets'], axis=1)



    @staticmethod
    def is_number(val):
        try:
            f=float(val)
            return True
        except ValueError:
            return False
    @staticmethod
    def getIdWithPaddingZeros(id):
        return str(int(id)).zfill(7)

    @staticmethod
    def getBase64EncodedVal(string_val):
        base64_bytes = base64.b64encode(string_val.encode('ascii'))
        # print("encode_data:", base64_bytes)
        return str(base64_bytes)[2:-1]

    @staticmethod
    def decodeBase64(str_bytes):
        bytes_64 = base64.b64decode(str_bytes)
        return str(bytes_64.decode('ascii'))

    @staticmethod
    def get_sha256(s):
        return sha256(s.encode('utf-8')).hexdigest()
    @staticmethod
    def uploadFileToS3(filepath,file_type="model"):
        # Generate a random file for the demo
        filename = os.path.split(filepath)[-1]
        # Define the API endpoint and the headers
        if file_type.lower() == "model":
            model_api_url = KGNET_Config.GML_ModelManager_URL + ":" + KGNET_Config.GML_ModelManager_PORT + "/model"
            filepath=filepath+".model" if filename.endswith(".model")==False else filepath
        elif file_type.lower() == "metadata":
            model_api_url = KGNET_Config.GML_ModelManager_URL + ":" + KGNET_Config.GML_ModelManager_PORT + "/metadata"
        headers = {"accept": "application/json"}
        # Perform the file upload
        with open(filepath, "rb") as f:
            response = requests.post(model_api_url, files={"model_file": f}, headers=headers)
        # Print the response from the server
        print(response.status_code)
        print(response.text)
        # Use the response to get the model's path in S3
        response_data = response.json()
        s3_path = response_data.get("s3_path")
        return s3_path

        # # Fetch the model from the server
        # get_url = model_api_url+f"/{filename}"
        # response_get = requests.get(get_url, headers=headers)
        # Print the status and save the retrieved file for verification if needed
        # print(response_get.status_code)
        # Print the content of the retrieved file
        # print("Content of the retrieved file:")
        # print(response_get.text)
        # return response_get.text


    def DownloadFileFromS3(filename,to_filepath,file_type="model"):
        # Define the API endpoint and the headers
        if file_type.lower()=="model":
            model_api_url = KGNET_Config.GML_ModelManager_URL + ":" + KGNET_Config.GML_ModelManager_PORT + "/model/"
            filename = filename.replace(".model","")
        elif file_type.lower() == "metadata":
            model_api_url = KGNET_Config.GML_ModelManager_URL + ":" + KGNET_Config.GML_ModelManager_PORT + "/metadata/"
        elif file_type.lower() == "emb":
            if not os.path.exists(KGNET_Config.emb_store_path):
                os.mkdir(KGNET_Config.emb_store_path)
            model_api_url = KGNET_Config.GML_ModelManager_URL + ":" + KGNET_Config.GML_ModelManager_PORT + "/emb_store/"
            filename = filename.replace(".model",".zip")


        headers = {"accept": "application/json"}
        # Perform the file download
        response = requests.get(model_api_url+f"{filename}", stream=True)
        if os.path.exists(to_filepath):
            os.remove(to_filepath)
        try:
            filename = stream.stream_response_to_file(response,path=to_filepath)
        except exceptions.StreamingError as e:
            print(e.message)
            return False
        return True

    @staticmethod
    def DAGTraversalsDFS(graph, start_node, current_path, all_paths):
        current_path.append(start_node)
        # print(f"current_path={current_path}")
        if len(current_path) == len(graph):
            all_paths.append(current_path.copy())
        neighbours_lst = graph[start_node]
        if len(neighbours_lst) < 1:
            return current_path
        elif len(neighbours_lst) == 1:
            return utils.DAGTraversalsDFS(graph, neighbours_lst[0], current_path, all_paths)
        elif len(neighbours_lst) > 1:
            perms = list(itertools.permutations(neighbours_lst))
            perm_paths = []
            for perm_neighbours in perms:
                # print(f"perm_neighbours={perm_neighbours}")
                perm_path = current_path.copy()
                ret_perm_paths = []
                # print(f"perm_path={perm_path}")
                for neighbor in perm_neighbours:
                    # print(f"neighbor={neighbor}")
                    if not any(isinstance(el, list) for el in ret_perm_paths):  # not list of lists
                        ret_perm_paths = utils.DAGTraversalsDFS(graph, neighbor, perm_path, all_paths)
                    else:
                        for ret_prem_path in ret_perm_paths: #multi path exist under this node
                            ret_perm_path = utils.DAGTraversalsDFS(graph, neighbor, ret_prem_path, all_paths)
                perm_paths.append(perm_path.copy())
            return perm_paths

    @staticmethod
    def generateAllDAGTravesals(edges_list=None, start_node='NC1', addStartNode=False,removeStartNode=False):
        # Define the graph edges as given
        # edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (4, 8), (2, 6), (3, 7), (7, 9)]
        if edges_list is not None:
            edges=edges_list
        else:
            edges = [('NC1', 'NC3'), ('NC2', 'NC4'), ('NC3', 'NC5')]
        if addStartNode:
            l = []
            list(l.extend(row) for row in edges)
            uniqueNode = set(l)
            start_nodes = []
            for n in uniqueNode:
                if sum([1 for (u, v) in edges if v == n]) == 0:
                    start_nodes.append(n)
            for n in start_nodes:
                edges.append((0, n))
            start_node = 0

        # Construct the graph as an adjacency list
        graph = {}
        for edge in edges:
            u, v = edge
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
        # Initialize variables for DFS
        # start_node = 0
        current_path = []
        all_paths = []
        # Generate all possible execution plans using DFS
        utils.DAGTraversalsDFS(graph, start_node, current_path, all_paths)
        # Sort Generated Paths
        all_paths = ['->'.join([str(le) for le in elem]) for elem in all_paths]
        all_paths.sort()
        if addStartNode or removeStartNode:
            all_paths = [elem.split('->')[1:] for elem in all_paths]
        else:
            all_paths = [elem.split('->') for elem in all_paths]
        # for path in all_paths:
        #     print(path)
        return all_paths

if __name__ == '__main__':
    ""
    # input="graph sainr->dblp->NC->03012024"
    # encoded_String=utils.getBase64EncodedVal(input)
    # print("encoded_String=",encoded_String)
    # decoded_String=utils.decodeBase64(encoded_String)
    # print("decoded_String=", decoded_String)
    # print(utils.get_sha256(input))
    # utils.uploadFileToS3("/home/hussein/Downloads/HMP.pdf",file_type="metadata")
    # utils.DownloadFileFromS3("mid-4d7c0825f06b2e2fea2866d2ec9e97fca4422649127983dde2a409020b5abadb","/home/hussein/Downloads/4d7c0825f06b2e2fea2866d2ec9e97fca4422649127983dde2a409020b5abadb.model",file_type="model")