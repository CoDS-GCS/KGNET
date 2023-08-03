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
    datasets_output_path="/media/hussein/UbuntuData/GithubRepos/KGNET/Datasets/"
    GML_API_URL = "http://127.0.0.1:64646/"
    KGMeta_IRI = "http://kgnet/"
    KGMeta_endpoint_url = "http://206.12.98.118:8890/sparql/"
    def __init__(self):
       ""

