import argparse
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import gzip
import datetime
import os
import shutil
import itertools
import random
from sklearn.metrics import precision_recall_fscore_support as score
import gc
import copy
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing
import re


def compress_gz(f_path):
    f_in = open(f_path, 'rb')
    f_out = gzip.open(f_path + ".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
def write_entity_mapping(item):
    key,val,output_root_path,dataset_name=item
    val = pd.DataFrame(list(val), columns=['ent name']).astype(
        'str').sort_values(by="ent name").reset_index(drop=True)
    val = val.drop_duplicates()
    val["ent idx"] = val.index
    val = val[["ent idx", "ent name"]]
    dic_res = pd.Series(val["ent idx"].values,index=val["ent name"]).to_dict()
    # print("key=",entites_dic[key+"_dic"])
    map_folder = output_root_path + dataset_name + "/mapping"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    val.to_csv(map_folder + "/" + key + "_entidx2name.csv.gz", index=None,compression='gzip')
    # compress_gz(map_folder + "/" + key + "_entidx2name.csv")
    return (key, val,dic_res )


def remove_special_characters(string):
    # Define the pattern for special characters
    pattern = r'.*?([@:\/].*)'

    # Find the match
    match = re.match(pattern, string)

    if match:
        # Extract the characters after the special character
        return match.group(1)
    else:
        # No special character found, return the original string
        return string
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    filtered = re.sub(pattern, '', text)
    return filtered

def write_relations_mapping(item):
    to_remove=None
    relations_entites_map_rel,relations_dic_rel,entites_dic,output_root_path,dataset_name,rel_idx=item
    for rel_list in relations_entites_map_rel:
        e1, rel, e2 = rel_list
        # ('human', 'P1038', 'Unknown_Entity')
        temp_relations_dic = copy.deepcopy(relations_dic_rel)
        temp_relations_dic["s_idx"] = temp_relations_dic["s"]  # .apply(
        # lambda x: str(x).split("/")[-1])
        temp_relations_dic["s_idx"] = temp_relations_dic["s_idx"].apply(
            lambda x: entites_dic[e1 + "_dic"][x] if x in entites_dic[
                e1 + "_dic"].keys() else -1)
        temp_relations_dic = temp_relations_dic[temp_relations_dic["s_idx"] != -1]
        ################
        # relations_dic[rel]["o_keys"]=relations_dic[rel]["o"].apply(lambda x:x.split("/")[3] if x.startswith("http") and len(x.split("/")) > 3 else x)
        temp_relations_dic["o_idx"] = temp_relations_dic["o"]
        temp_relations_dic["o_idx"] = temp_relations_dic["o_idx"].apply(
            lambda x: entites_dic[e2 + "_dic"][x] if x in entites_dic[
                e2 + "_dic"].keys() else -1)
        temp_relations_dic = temp_relations_dic[temp_relations_dic["o_idx"] != -1]

        temp_relations_dic = temp_relations_dic.sort_values(by="s_idx").reset_index(drop=True)
        rel_out = temp_relations_dic[["s_idx", "o_idx"]]
        if len(rel_out) > 0:
            map_folder = output_root_path + dataset_name + "/raw/relations/" + e1 + "___" + \
                         rel.split("/")[-1] + "___" + e2
            try:
                os.stat(map_folder)
            except:
                os.makedirs(map_folder)
            rel_out.to_csv(map_folder + "/edge.csv.gz", index=None, header=None,compression='gzip')
            # compress_gz(map_folder + "/edge.csv")
            ########## write relations num #################
            # f = open(map_folder + "/num-edge-list.csv", "w")
            # f.write(str(len(temp_relations_dic)))
            # f.close()
            # compress_gz(map_folder + "/num-edge-list.csv")
            pd.DataFrame({0: [len(temp_relations_dic)]}).to_csv(os.path.join(map_folder, "num-edge-list.csv.gz"),
                                                                header=None, index=None, compression='gzip')
            ##################### write relations idx #######################
            rel_out["rel_idx"] = rel_idx
            rel_idx_df = rel_out["rel_idx"]
            rel_idx_df.to_csv(map_folder + "/edge_reltype.csv.gz", header=None, index=None, compression='gzip')
            # compress_gz(map_folder + "/edge_reltype.csv")
        else:
            to_remove=[e1, str(rel).split("/")[-1], e2]
    return to_remove
def encode_entities(item):
    relations_lst,g_tsv_df,g_tsv_types_df,len_entities_isa_df,entities_isa_dict=item
    relations_entites_map = {}
    relations_dic = {}
    entites_dic = {}
    for rel in tqdm(relations_lst):
        rel_type = rel.split("/")[-1]
        # rel_type = rel812
        rel_df = g_tsv_df[g_tsv_df["p"] == rel_type].reset_index(drop=True)
        # print("rel_type ", rel)
        list_rel_types = []
        if len_entities_isa_df > 0:
            rel_df["s_type"] = rel_df["s"].apply(lambda x:entities_isa_dict[x] if str(x) in entities_isa_dict.keys() else "Subject-"+rel_type)
            rel_df["o_type"] = rel_df["o"].apply(lambda x: entities_isa_dict[x] if str(x) in entities_isa_dict.keys() else "Object-"+rel_type)
        else:
            rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([rel_type])]
            s_type = rel_types['stype'].values[0]
            o_type = rel_types['otype'].values[0]
            rel_df["s_type"] = s_type
            rel_df["o_type"] = o_type

        for idx,row in rel_df[["s_type","o_type"]].drop_duplicates().iterrows():
            list_rel_types.append((row["s_type"], rel, row["o_type"]))
        relations_entites_map[rel] = list_rel_types
            # if len(list_rel_types) > 2:
                # print(len(list_rel_types))
        relations_dic[rel] = rel_df
        # e1_list=list(set(relations_dic[rel]["s"].apply(lambda x:str(x).split("/")[:-1])))
        for rel_pair in list_rel_types:
            e1, rel, e2 = rel_pair
            # if e1 == "human":
            #     print("e1=", e1)

            if e1 != "literal" and e1 in entites_dic:
                entites_dic[e1] = entites_dic[e1].union(
                    set(rel_df[rel_df["s_type"] == e1]["s"].unique()))  # .apply(
                # lambda x: str(x).split("/")[-1]).unique()))
            elif e1 != "literal":
                entites_dic[e1] = set(rel_df[rel_df["s_type"] == e1]["s"].unique())  # .apply(
                # lambda x: str(x).split("/")[-1]).unique())

            if e2 != "literal" and e2 in entites_dic:
                entites_dic[e2] = entites_dic[e2].union(
                    set(rel_df[rel_df["o_type"] == e2]["o"].unique()))  # .apply(
                # lambda x: str(x).split("/")[-1]).unique()))
            elif e2 != "literal":
                entites_dic[e2] = set(rel_df[rel_df["o_type"] == e2]["o"].unique())  # .apply(
                # lambda x: str(x).split("/")[-1]).unique())
    return (entites_dic,relations_entites_map,relations_dic)
###################### Zip Folder to OGB Format
# zip -r mag_ComputerProgramming_papers_venue_QM3.zip mag_ComputerProgramming_papers_venue_QM3/ -i '*.gz'
def define_rel_types(g_tsv_df):
    g_tsv_df["p"]

def transform_tsv_to_PYG(dataset_name,dataset_name_csv,dataset_types,split_rel,target_rel,similar_target_rels,output_root_path
                         ,MINIMUM_INSTANCE_THRESHOLD=21,test_size=0.1,valid_size=0.1,split_rel_train_value=None,split_rel_valid_value=None,Header_row=None,targetNodeType=None,labelNodetype=None,inference=False,nthreads=6):
    dic_results = {}  # args.dic_results #{}
    start_t = datetime.datetime.now()
    if dataset_types == "":
        dataset_types = output_root_path + dataset_name_csv + "_types.csv"

    if Header_row is None:
        g_tsv_df = pd.read_csv(output_root_path + dataset_name_csv + ".tsv", encoding_errors='ignore', sep="\t",header=Header_row)
        g_tsv_types_df = pd.read_csv(dataset_types, encoding_errors='ignore',header=Header_row)
        g_tsv_df.columns=["s","p","o"]
        g_tsv_types_df.columns=['stype','ptype', 'otype']
        ############filter less representiaive relations <6 instances ################
        if not inference:
            p_counts_dict = g_tsv_df["p"].value_counts().to_dict()
            repesentative_p_lst= [k for k, v in p_counts_dict.items() if v > 6]
            g_tsv_df=g_tsv_df[g_tsv_df["p"].isin(repesentative_p_lst)]
    else:
        g_tsv_df = pd.read_csv(output_root_path + dataset_name_csv + ".tsv", encoding_errors='ignore', sep="\t",header=Header_row,names=['s','p','o'])
        g_tsv_types_df = pd.read_csv(dataset_types, encoding_errors='ignore',header=None,names=['stype','ptype','otype'])
    #################### Remove '/' from the predicate of the Graph (tsv) ##############################
    g_tsv_df["p"] = g_tsv_df["p"].apply(lambda x: x.split('/')[-1].split("#")[-1].split(":")[-1])
    target_rel = target_rel.split('/')[-1].split("#")[-1].split(':')[-1]
    # target_rel = remove_special_characters(target_rel)
    ############### special case for DBLP PV Task ###############3
    if dataset_types .split("/")[-1] in ["dblp2022_Types.csv","dblp_Types.csv"] and target_rel == "publishedIn":
        similar_target_rels.append("publishedInJournal")
        similar_target_rels.append("publishedInBook")
    print(f"similar_target_rels={similar_target_rels}")
    ##################Check if headers are present in _types file
    if any(col not in g_tsv_types_df.columns for col in ['stype', 'ptype', 'otype']):
        old_columns = g_tsv_types_df.columns
        # print('***OLD g_tsv_types df:',g_tsv_types_df.head())
        # print(f'changing "{old_columns[0]}" to stype, "{old_columns[1]}" to ptype , and "{old_columns[2]}" to otype !')
        g_tsv_types_df = g_tsv_types_df.rename(
            columns={old_columns[0]: 'stype', old_columns[1]: 'ptype', old_columns[2]: 'otype'})
        # print('***New g_tsv_types df:',g_tsv_types_df.head())
    # print("original_g_csv_df loaded , records length=", len(g_tsv_df))

    # dataset_name += "_Discipline"
    try:
        g_tsv_df = g_tsv_df.rename(columns={"Subject": "s", "Predicate": "p", "Object": "o"})
        g_tsv_df = g_tsv_df.rename(columns={0: "s", 1: "p", 2: "o"})
        ######################## Remove Litreal Edges####################
        Literal_edges_lst = []
        g_tsv_df = g_tsv_df[~g_tsv_df["p"].isin(Literal_edges_lst)]
        # print("len of g_tsv_df after remove literal edges types ", len(g_tsv_df))
        g_tsv_df = g_tsv_df.drop_duplicates()
        # print("len of g_tsv_df after drop_duplicates  ", len(g_tsv_df))
        g_tsv_df = g_tsv_df.dropna()
        # print("len of g_tsv_df after dropna  ", len(g_tsv_df))
    except:
        print("g_tsv_df columns=", g_tsv_df.columns())
    # unique_p_lst = g_tsv_df["p"].unique().tolist()
    ########################delete non target nodes #####################
    # if labelNodetype is not None:
    #     all_labelNodetype_lst = g_tsv_df[(g_tsv_df["p"] == "type") & (g_tsv_df["o"] == labelNodetype)]["s"].unique().tolist()
    #     all_target_nodes_set=set(g_tsv_df[g_tsv_df["p"] == target_rel]["s"].unique().tolist())
    #     all_labelNodetype_target_nodes_set = set(g_tsv_df[(g_tsv_df["p"] == target_rel) &(g_tsv_df["o"].isin(all_labelNodetype_lst))]["s"].unique().tolist())
    #     non_target_nodes_set=all_target_nodes_set-all_labelNodetype_target_nodes_set
    #     g_tsv_df=g_tsv_df[~g_tsv_df["s"].isin(non_target_nodes_set)]
    #     g_tsv_df=g_tsv_df[~g_tsv_df["o"].isin(non_target_nodes_set)]

    relations_lst = g_tsv_df["p"].unique().astype("str").tolist()
    relations_lst = [rel for rel in relations_lst if rel not in similar_target_rels]
    # print("relations_lst=", relations_lst)
    dic_results["usecase"] = dataset_name
    dic_results["TriplesCount"] = len(g_tsv_df)

    #################### Remove Split and Target Rel ############
    # if split_rel in relations_lst:
    #     relations_lst.remove(split_rel)
    if target_rel in relations_lst:
        relations_lst.remove(target_rel)
    for srel in ["type","label"]:
        if srel in relations_lst:
            relations_lst.remove(srel)
    for srel in similar_target_rels:
        if srel in relations_lst:
            relations_lst.remove(srel)
    ################################Start Encoding Nodes and edges ########################
    ################################write relations index ########################
    relations_df = pd.DataFrame(relations_lst, columns=["rel name"])
    relations_df["rel name"] = relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
    relations_df["rel idx"] = relations_df.index
    relations_df = relations_df[["rel idx", "rel name"]]
    map_folder = output_root_path + dataset_name + "/mapping"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    relations_df.to_csv(map_folder + "/relidx2relname.csv.gz", index=None, compression='gzip')
    # compress_gz(map_folder + "/relidx2relname.csv")
    ###########################################prepare entities encoding#################################
    relations_entites_map = {}
    relations_dic = {}
    entites_dic = {}
    entities_isa_df=g_tsv_df[g_tsv_df["p"] == "type"]
    entities_isa_df=entities_isa_df.drop_duplicates()
    ############## replace Special characters for files naming ###################
    special_chars_lst=['\n','/',':','\"','>','<','\\','|','?','*','.'] # chars not allowed in file name.
    otypes_unique_lst=entities_isa_df["o"].unique().tolist()
    otypes_unique_dic={}
    for entity_type in otypes_unique_lst:
        otypes_unique_dic[entity_type]=''.join(['-' if s in special_chars_lst else s for s in entity_type.split("/")[-1].split("#")[-1]]).strip()
    entities_isa_df["o"] = entities_isa_df["o"].apply(lambda x: otypes_unique_dic[str(x)])

    ############################### Obtain Target node if not explicitly given #########
    if targetNodeType is None:
        targetNodeType = list(g_tsv_types_df[g_tsv_types_df['ptype'] == target_rel.split("/")[-1].split("#")[-1].split(":")[-1]]['stype'].value_counts().to_dict().keys())[0]
    else:
        targetNodeType=targetNodeType.split("/")[-1].split("#")[-1].split(":")[-1]
    ##################### delete other types for target Node that has multi-type ###################
    target_nodes_lst=entities_isa_df[entities_isa_df["o"]==targetNodeType]["s"].unique().tolist()
    to_delete_target_node_types=entities_isa_df[(entities_isa_df["s"].isin(target_nodes_lst)) & (entities_isa_df["o"] != targetNodeType)]
    entities_isa_df = entities_isa_df.drop(index=to_delete_target_node_types.index.to_list())
    entities_isa_df = entities_isa_df.reset_index()
    ###############################################################################################
    if len(entities_isa_df)>0:
        entities_isa_dict=dict(zip(list(entities_isa_df.s), list(entities_isa_df.o)))
    ###########################################################
    print("Encode Entities of Relations")
    chunksize=int(len(relations_lst)/nthreads)
    chunksize= 5 if chunksize<5 else chunksize
    chunks=[relations_lst[i:(i+chunksize)] for i in range(0,len(relations_lst),chunksize)]
    with multiprocessing.Pool(nthreads) as pool:
        items=[(chunk,g_tsv_df, g_tsv_types_df, len(entities_isa_df), entities_isa_dict) for chunk in chunks]
        res=tqdm(pool.imap(encode_entities,items),total=len(items))
        pool.close()
        pool.join()

    # items = [(chunk, g_tsv_df, g_tsv_types_df, len(entities_isa_df), entities_isa_dict) for chunk in chunks]
    # from concurrent.futures import ThreadPoolExecutor,as_completed,ProcessPoolExecutor
    # with ThreadPoolExecutor(max_workers = os.cpu_count()*2) as executor:
    #     futures = [executor.submit(encode_entities, item) for item in items]
    #     res = []
    #     for future in tqdm(as_completed(futures), total=len(items)):
    #         res.append(future.result())

    ############ Union Mappings ##########
    for (ch_entites_dic,ch_relations_entites_map,ch_relations_dic) in res:
        for key in ch_entites_dic.keys():
            if key in entites_dic:
                entites_dic[key]=entites_dic[key].union(ch_entites_dic[key])
            else:
                entites_dic[key] = ch_entites_dic[key]
        for key in ch_relations_entites_map.keys():
            if key in relations_entites_map:
                relations_entites_map[key]=relations_entites_map[key].union(ch_relations_entites_map[key])
            else:
                relations_entites_map[key] = ch_relations_entites_map[key]
        for key in ch_relations_dic.keys():
            if key in relations_dic:
                relations_dic[key]=relations_dic[key].union(ch_relations_dic[key])
            else:
                relations_dic[key] = ch_relations_dic[key]


    ############################### Make sure all target nodes have label ###########
    # target_subjects_lst = g_tsv_df[g_tsv_df["p"] == target_rel]["s"].unique().tolist()
    # entites_dic[targetNodeType] = set.intersection(entites_dic[targetNodeType], set(target_subjects_lst))
    ############################ write entites index #################################
    print("write entites mappings")
    # # for key in tqdm(list(entites_dic.keys())):
    with multiprocessing.Pool(nthreads) as pool:
        # tqdm(pool.imap(write_entity_mapping, list(entites_dic.keys())), total=len(list(entites_dic.keys())))
        items=[(key,entites_dic[key],output_root_path,dataset_name) for key in list(entites_dic.keys())]
        res=tqdm(pool.imap(write_entity_mapping,items),total=len(items))
        pool.close()
        pool.join()

    # items = [(key, entites_dic[key], output_root_path, dataset_name) for key in list(entites_dic.keys())]
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(write_entity_mapping,item) for item in items]
    #     res = []
    #     for future in tqdm(as_completed(futures), total=len(items)):
    #         res.append(future.result())

    for (key,val1,val2) in res:
        # print(key)
        entites_dic[key]=val1
        entites_dic[key+"_dic"] = val2
    #################### write nodes statistics ######################
    lst_node_has_feat = [list(filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
    lst_node_has_label = lst_node_has_feat.copy()
    lst_num_node_dict = lst_node_has_feat.copy()
    lst_has_feat = []
    lst_has_label = []
    lst_num_node = []

    for entity in lst_node_has_feat[0]:
        if str(entity) == str(targetNodeType):
            lst_has_label.append("True")
            lst_has_feat.append("True")
        else:
            lst_has_label.append("False")
            lst_has_feat.append("False")

        # lst_has_feat.append("False")
        lst_num_node.append(len(entites_dic[entity + "_dic"]))

    lst_node_has_feat.append(lst_has_feat)
    lst_node_has_label.append(lst_has_label)
    lst_num_node_dict.append(lst_num_node)
    lst_relations = []

    for key in list(relations_entites_map.keys()):
        for elem in relations_entites_map[key]:
            (e1, rel, e2) = elem
            lst_relations.append([e1, str(rel).split("/")[-1], e2])

    map_folder = output_root_path + dataset_name + "/raw"
    # print("map_folder=", map_folder)
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)

    pd.DataFrame(lst_node_has_feat).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-feat.csv.gz", header=None,
        index=None, compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-feat.csv")
    pd.DataFrame(lst_node_has_label).to_csv(output_root_path + dataset_name + "/raw/nodetype-has-label.csv.gz",header=None, index=None, compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-label.csv")
    pd.DataFrame(lst_num_node_dict).to_csv(output_root_path + dataset_name + "/raw/num-node-dict.csv.gz", header=None,index=None, compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/num-node-dict.csv")
    ############################### create labels index ########################
    target_labels_df = g_tsv_df[g_tsv_df["p"] == target_rel]
    target_labels_df = target_labels_df[target_labels_df["s"].isin(entites_dic[targetNodeType+"_dic"])]
    if not inference:
        representative_labels_lst=[k for k,v in target_labels_df["o"].value_counts().to_dict().items() if v>=MINIMUM_INSTANCE_THRESHOLD]
    else:
        representative_labels_lst = [k for k, v in target_labels_df["o"].value_counts().to_dict().items()]
    target_labels_df=target_labels_df[target_labels_df["o"].isin(representative_labels_lst)]
    if labelNodetype is None:
        label_idx_df = pd.DataFrame(target_labels_df["o"].apply(lambda x: str(x).strip()).unique().tolist(),
                                    columns=["label name"])
    else:
        all_labels_df = g_tsv_df[g_tsv_df["s"].isin(target_labels_df["o"])]
        label_idx_df = pd.DataFrame(
            all_labels_df[(all_labels_df["p"] == "type") & (all_labels_df["o"] == labelNodetype)]["s"].apply(
                lambda x: str(x).strip()).unique().tolist(), columns=["label name"])
    dic_results["ClassesCount"] = len(label_idx_df)
    try:
        label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
    except:
        label_idx_df["label name"] = label_idx_df["label name"].astype("str")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)

    label_idx_df["label idx"] = label_idx_df.index
    label_idx_df = label_idx_df[["label idx", "label name"]]
    label_idx_df.to_csv(output_root_path + dataset_name + "/mapping/labelidx2labelname.csv.gz", index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/mapping/labelidx2labelname.csv")
    ############################### create label relation index  ######################
    label_idx_df["label idx"] = label_idx_df["label idx"].astype("int64")
    # label_idx_df["label name"] = label_idx_df["label name"].apply(lambda x: str(x).split("/")[-1])
    label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
    ############ drop multiple targets per subject keep first#######################
    if labelNodetype is None:
        labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel].reset_index(drop=True)
    else:
        labels_rel_df = g_tsv_df[(g_tsv_df["p"] == target_rel)&(g_tsv_df["o"].isin(label_idx_df["label name"]))].reset_index(drop=True)
    labels_rel_df = labels_rel_df.sort_values(['s', 'o'], ascending=[True, True])
    labels_rel_df = labels_rel_df.drop_duplicates(subset=["s"], keep='first')
    ###############################################################################
    target_without_labels_set = set(entites_dic[targetNodeType + "_dic"]) - set(labels_rel_df["s"].tolist())
    traget_with_dummy_label_dic = {(entites_dic[targetNodeType + "_dic"][elem] if elem in entites_dic[
            targetNodeType + "_dic"].keys() else -1): -1 for elem in target_without_labels_set}
    traget_with_dummy_label_df=pd.DataFrame({'s_idx':traget_with_dummy_label_dic.keys(),'o_idx':traget_with_dummy_label_dic.values()})
    label_type = targetNodeType
    labels_rel_df["s_idx"] = labels_rel_df["s"]
    labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("str")
    labels_rel_df["s_idx"] = labels_rel_df["s_idx"].apply(
        lambda x: entites_dic[targetNodeType + "_dic"][x] if x in entites_dic[
            targetNodeType + "_dic"].keys() else -1)
    labels_rel_df = labels_rel_df[labels_rel_df["s_idx"] != -1]
    labels_rel_df["o_idx"] = labels_rel_df["o"]  # .apply(lambda x: str(x).split("/")[-1])
    labels_rel_df["o_idx"] = labels_rel_df["o_idx"].apply(
        lambda x: label_idx_dic[str(x)] if str(x) in label_idx_dic.keys() else -1)
    labels_rel_df=labels_rel_df[["s_idx","o_idx"]]
    labels_rel_df=pd.concat([labels_rel_df,traget_with_dummy_label_df]) ## append dummy target nodes
    labels_rel_df = labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)

    out_labels_df = labels_rel_df[["o_idx"]]
    map_folder = output_root_path + dataset_name + "/raw/node-label/" + targetNodeType
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    out_labels_df.to_csv(map_folder + "/node-label.csv.gz", header=None, index=None,compression='gzip')
    # compress_gz(map_folder + "/node-label.csv")
    ###########################################split parts (train/test/validate)#########################
    # split_df = g_tsv_df[g_tsv_df["p"] == split_rel]
    map_folder = output_root_path + dataset_name + "/split/" + label_type  # + split_by[        "folder_name"] + "/"
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    if not inference:
        print("Split Dataset into Train/Valid/Test")
        if split_rel is None:
            split_rel = 'random'
        split_rel = split_rel.split('/')[-1]
        if split_rel.lower() == 'random':
            if labelNodetype is None:
                split_df = g_tsv_df[(g_tsv_df["p"] == target_rel) & (g_tsv_df["o"].isin(label_idx_df["label name"]))]
            else:
                split_df = g_tsv_df[g_tsv_df["p"] == target_rel]
        else:
            if labelNodetype is None:
                split_df = g_tsv_df[(g_tsv_df["p"] == split_rel) & (g_tsv_df["o"].isin(label_idx_df["label name"]))]
            else:
                split_df = g_tsv_df[g_tsv_df["p"] == split_rel]
        ####################### filter for traget node types only ###############
        split_df = split_df[split_df["s"].isin(entites_dic[targetNodeType+"_dic"])]
        split_df = split_df[~split_df["s"].isin(target_without_labels_set)] ## exclude dummy label target nodes
        ########## remove target node  with multi labels ################
        target_label_dict = split_df["s"].value_counts().to_dict()
        target_nodes_to_keep_lst = list(k for k, v in target_label_dict.items() if v == 1)
        split_df = split_df[split_df["s"].isin(target_nodes_to_keep_lst)]
        ########## remove labels with less than MINIMUM_INSTANCE_THRESHOLD samples################
        labels_dict = split_df["o"].value_counts().to_dict()
        if not inference:
            labels_to_keep_lst = list(k for k, v in labels_dict.items() if v >= MINIMUM_INSTANCE_THRESHOLD)
            split_df = split_df[split_df["o"].isin(labels_to_keep_lst)]
        #############################################################
        split_df["s"] = split_df["s"].astype("str").apply(lambda x: entites_dic[targetNodeType + "_dic"][str(x)] if x in entites_dic[
            targetNodeType + "_dic"] else -1)

        split_df = split_df[split_df["s"] != -1]
        label_type_values_lst = list(entites_dic[label_type + "_dic"].values())
        split_df = split_df[split_df["s"].isin(label_type_values_lst)]
        split_df = split_df.sort_values(by=["s"]).reset_index(drop=True)

        # train_df = split_df[split_df["o"] <= split_by["train"]]["s"]
        # valid_df = split_df[(split_df["o"] > split_by["train"]) & (split_df["o"] <= split_by["valid"])]["s"]
        # test_df = split_df[(split_df["o"] > split_by["valid"])]["s"]

        ########## Random Splitting ###########
        if split_rel is None or split_rel.lower() == 'random':
            dic_results["testSize"] = test_size
            dic_results["validSize"] = valid_size
            dic_results["trainSize"] = 1 - (test_size+valid_size)
            X_train, X_test, y_train, y_test = train_test_split(split_df["s"].tolist(), split_df["o"].tolist(),
                                                                test_size=test_size+valid_size, random_state=42,
                                                                stratify=split_df["o"].tolist())
            try:
                X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=valid_size/(test_size+valid_size), random_state=42,stratify=y_test)
            except:
                X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=valid_size/(test_size+valid_size), random_state=42)
            train_df = pd.DataFrame(X_train)
            valid_df = pd.DataFrame(X_valid)
            test_df = pd.DataFrame(X_test)
        else:
                # print("SPLIT REL PROVIDED: ", split_rel)
                dic_results["splitEdge"]=split_rel
                train_df = split_df[split_df["o"].astype(int) <= split_rel_train_value]["s"]
                valid_df = split_df[(split_df["o"].astype(int) > split_rel_train_value) & (
                            split_df["o"].astype(int) <= split_rel_valid_value)]["s"]
                test_df = split_df[(split_df["o"].astype(int) > split_rel_valid_value)]["s"]


        train_df.to_csv(map_folder + "/train.csv.gz", index=None, header=None,compression='gzip')
        # compress_gz(map_folder + "/train.csv")
        valid_df.to_csv(map_folder + "/valid.csv.gz", index=None, header=None,compression='gzip')
        # compress_gz(map_folder + "/valid.csv")
        test_df.to_csv(map_folder + "/test.csv.gz", index=None, header=None,compression='gzip')
        # compress_gz(map_folder + "/test.csv")
    ###################### create nodetype-has-split.csv#####################
    lst_node_has_split = [
        list(
            filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
    lst_has_split = []
    for rel in lst_node_has_split[0]:
        if rel == label_type:
            lst_has_split.append("True")
        else:
            lst_has_split.append("False")
    lst_node_has_split.append(lst_has_split)
    pd.DataFrame(lst_node_has_split).to_csv(
        output_root_path + dataset_name + "/split/"  # + split_by["folder_name"]
        + "nodetype-has-split.csv.gz", header=None, index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/split/"  # + split_by["folder_name"]
                # + "nodetype-has-split.csv")
    ###################### write entites relations for nodes only (non literals) #########################
    idx = 0
    print("write entites relations")
    with multiprocessing.Pool(nthreads) as pool:
        # # tqdm(pool.imap(write_entity_mapping, list(entites_dic.keys())), total=len(list(entites_dic.keys())))
        items = [(relations_entites_map[rel], relations_dic[rel], entites_dic, output_root_path, dataset_name,  relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]) for rel in relations_dic.keys()]
        res = tqdm(pool.imap(write_relations_mapping, items), total=len(items))
        pool.close()
        pool.join()
    for res_key in res:
        if res_key is not None and res_key in lst_relations:
            lst_relations.remove(res_key)

    # items = [(relations_entites_map[rel], relations_dic[rel], entites_dic, output_root_path, dataset_name,
    #           relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]) for rel in
    #          relations_dic.keys()]
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(write_relations_mapping,item) for item in items]
    #     res = []
    #     for future in tqdm(as_completed(futures), total=len(items)):
    #         res.append(future.result())
    for res_key in res:
        if res_key is not None and res_key in lst_relations:
            lst_relations.remove(res_key)



    # items = [(relations_entites_map[rel], relations_dic[rel], entites_dic, output_root_path, dataset_name,  relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]) for rel in relations_dic.keys()]
    # for item in tqdm(items):
    #     res = write_relations_mapping(item)
    #     if res is not None and res in lst_relations:
    #         lst_relations.remove(res)

    pd.DataFrame(lst_relations).to_csv(
        output_root_path + dataset_name + "/raw/triplet-type-list.csv.gz",
        header=None, index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/triplet-type-list.csv")
    #####################Zip Folder ###############
    shutil.make_archive(output_root_path + dataset_name, 'zip',
                            root_dir=output_root_path, base_dir=dataset_name)
    end_t = datetime.datetime.now()
    dic_results["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()
    print(dic_results)
    # print(dataset_name.split(".")[0] + "_csv_to_Hetrog_time=", end_t - start_t, " sec.")
    dic_results['label_mapping'] = {v: k for k, v in label_idx_dic.items()}  # {v : k for k,v in entites_dic[targetNodeType+'_dic'].items()}
    # pd.DataFrame(dic_results).to_csv(
    #     output_root_path + dataset_name.split(".")[0] + "_PYG_Transformation_times.csv", index=False)
    return dic_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSV to PYG')
    #parser.add_argument('--csv_path', type=str, default="")
    # parser.add_argument('--target_rel', type=str, default="http://www.wikidata.org/entity/P27") # https://dblp.org/rdf/schema#publishedIn
    parser.add_argument('--target_rel', type=str, default="http://www.wikidata.org/entity/P106")  # https://dblp.org/rdf/schema#publishedIn
    #split_data_type
    #size of train # if random then in % else specific value for train and valid split
    #size of valid
    parser.add_argument('--dataset_name',type=str, default="WikiKG2015V2_22MT_FG_human_106_occupation_NC") # name of generated zip file
    #parser.add_argument('--dataset_csv',type=str, default="")
    parser.add_argument('--dataset_name_csv',type=str, default="WikiKG")  # csv/tsv , input dataset name
    parser.add_argument('--dataset_types',type=str, default="/media/hussein/WindowsData/WikiKG_V2_2015/WikiKG2015_v2_Types.csv") # path to the 'types' file containing relatiolns
    #parser.add_argument('--split_by', type=dict, default={}) replaced by train and valid args
    parser.add_argument('--split_rel', type=str, default="random") # https://dblp.org/rdf/schema#yearOfPublication #default = random # TODO could be null in some rows
    parser.add_argument('--split_rel_train_value', type=int, default=0)
    parser.add_argument('--split_rel_valid_value', type=int, default=0)
    parser.add_argument('--test_size',type=float,default=0.2)
    parser.add_argument('--valid_size',type=float,default=0.5)
    parser.add_argument('--similar_target_rels',type=list, default=[]) # 'https://dblp.org/rdf/schema#publishedInSeries' 
    # parser.add_argument('--targetNodeType',type=str, default = "")
    #parser.add_argument('--dic_results',type=dict, default = "")
    parser.add_argument('--Literals2Nodes',type=bool, default = False) # convert literal vals into nodes (eg name of paper )or ignore 
    parser.add_argument('--output_root_path',type=str, default = "/media/hussein/WindowsData/WikiKG_V2_2015/")
    parser.add_argument('--MINIMUM_INSTANCE_THRESHOLD',type=int, default = 21)

    args = parser.parse_args()
    dataset_name = args.dataset_name  # "biokg_Drug_Classification" # Name of the dataset
    dataset_name_csv = args.dataset_name_csv  # "biokg"  # spo in IRI .csv no need for <>
    dataset_types = args.dataset_types  # "biokg_types.csv"  # kind of ontology
    split_rel = args.split_rel  # "http://purl.org/dc/terms/year"
    # split_by = args.split_by #{"folder_name": "random"}  # , "split_data_type": "int", "train":2006  ,"valid":2007 , "test":2008 }
    target_rel = args.target_rel  # "https://www.biokg.org/CLASS"  # is in the dataset and is StudiedDrug
    similar_target_rels = args.similar_target_rels  # ["https://www.biokg.org/SUBCLASS", "https://www.biokg.org/SUPERCLASS"]
    Literals2Nodes = args.Literals2Nodes  # False
    output_root_path = args.output_root_path  # "/home/ubuntu/flora_tests/biokg/data/"
    transform_tsv_to_PYG(dataset_name,dataset_name_csv,dataset_types,split_rel,target_rel,similar_target_rels,output_root_path,args.MINIMUM_INSTANCE_THRESHOLD,args.test_size,args.valid_size,args.split_rel_train_value,args.split_rel_valid_value,Header_row=None,targetNodeType="human",labelNodetype="occupation")



    
