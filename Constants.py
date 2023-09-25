class GNN_Methods:
    Graph_SAINT = "G_SAINT"
    RGCN = "RGCN"
    ShaDowGNN = "ShaDowSAINT"
    SeHGGN = "SeHGNN"
    IBS = "IBS"
    MorsE = "MorsE"
    NodePiece = "NodePiece"
    LHGNN = "LHGNN"
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
    prefix = GNN_KG_TASK_PREFIX + 'KGPrefix'
    modelId = GNN_TASK_PREFIX + 'modelID'
    taskType = GNN_TASK_PREFIX + 'taskType'
    def __init__(self):
       ""
class GML_Query_Types:
    Inference = "Select"
    Insert = "Insert"
    Delete = "Delete"
    def __init__(self):
       ""
class TOSG_Patterns:
    d1h1 = "d1h1"
    d1h2 = "d1h2"
    d2h1 = "d2h1"
    d2h2 = "d2h2"
    def __init__(self):
       ""
class KGNET_Config:
    datasets_output_path="/home/afandi/GitRepos/KGNET/Datasets/"
    inference_path = datasets_output_path + 'Inference/'
    trained_model_path = datasets_output_path + 'trained_models/'
    GML_API_URL = "http://206.12.99.65:64647/"
    GML_Inference_PORT = "64647"
    GML_ModelManager_PORT = "64648"
    # GML_ModelManager_URL = "http://206.12.100.114"
    GML_ModelManager_URL = "http://206.12.99.65"
    KGMeta_IRI = "http://kgnet/"
    KGMeta_endpoint_url = "http://206.12.98.118:8890/sparql/"
    def __init__(self):
       ""


KGs_prefixs_dic={"dblp":"https://www.dblp.org/",
             "lkmdb":"https://www.lkmdb.org/",
             "mag":"https://www.mag.org/",
             "aifb":"http://www.aifb.uni-karlsruhe.de/"}
namedGraphURI_dic={"dblp":"http://dblp.org",
             "lkmdb":"https://www.lkmdb.org/",
             "mag":"https://www.mag.org/",
             "aifb":"http://www.aifb.uni-karlsruhe.de"}


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
    def getIdWithPaddingZeros(id):
        return str(int(id)).zfill(7)

