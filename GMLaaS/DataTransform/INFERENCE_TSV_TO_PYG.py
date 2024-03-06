import argparse
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import gzip
import datetime
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
from multiprocessing import Manager,Queue
import multiprocessing

from sklearn.model_selection import train_test_split


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


def process_relation(rel, g_tsv_df, g_tsv_types_df, entites_dic, relations_entites_map, relations_dic):
    try:
        rel_type = rel.split("/")[-1]
        rel_df = g_tsv_df[g_tsv_df["p"] == rel_type].reset_index(drop=True)
        rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([rel_type])]
        s_type = rel_types['stype'].values[0]
        o_type = rel_types['otype'].values[0]
        rel_df["s_type"] = s_type
        rel_df["o_type"] = o_type

        rel_entity_types = rel_df[["s_type", "o_type"]].drop_duplicates()
        list_rel_types = []
        for idx, row in rel_entity_types.iterrows():
            list_rel_types.append((row["s_type"], rel, row["o_type"]))

        relations_entites_map[rel] = list_rel_types
        relations_dic[rel] = rel_df
        for rel_pair in list_rel_types:
            e1, rel, e2 = rel_pair
            if e1 != "literal" and e1 in entites_dic:
                entites_dic[e1] = entites_dic[e1].union(set(rel_df[rel_df["s_type"] == e1]["s"].unique()))
            elif e1 != "literal":
                entites_dic[e1] = set(rel_df[rel_df["s_type"] == e1]["s"].unique())

            if e2 != "literal" and e2 in entites_dic:
                entites_dic[e2] = entites_dic[e2].union(set(rel_df[rel_df["o_type"] == e2]["o"].unique()))
            elif e2 != "literal":
                entites_dic[e2] = set(rel_df[rel_df["o_type"] == e2]["o"].unique())
        return
    except Exception as e:
        print(f"++++++++ DATA_TRANSFORMATION SKIPPING RELATION {rel}: {e} ++++++++")


def process_key(key, entites_dic, output_root_path, dataset_name):
    try:
        df = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype('str').sort_values(by="ent name").reset_index(drop=True)
        df = df.drop_duplicates()
        df["ent idx"] = df.index
        df = df[["ent idx", "ent name"]]
        entites_dic[key + "_dic"] = pd.Series(df["ent idx"].values, index=df["ent name"]).to_dict()

        map_folder = os.path.join(output_root_path, dataset_name, "mapping")
        os.makedirs(map_folder, exist_ok=True)
        df.to_csv(os.path.join(map_folder, f"{key}_entidx2name.csv.gz"), index=None,compression='gzip')
        # compress_gz(os.path.join(map_folder, f"{key}_entidx2name.csv"))

    except Exception as e:
        print(f"Error processing key {key}: {e}")

def write_relations(rel_list, relations_dic, entites_dic, relations_df, output_root_path, dataset_name):
    e1, rel, e2 = rel_list[0]
    relations_dic[rel]["s_idx"] = relations_dic[rel]["s"]
    relations_dic[rel]["s_idx"] = relations_dic[rel]["s_idx"].apply(
        lambda x: entites_dic[e1 + "_dic"][x] if x in entites_dic[e1 + "_dic"].keys() else -1)
    relations_dic[rel] = relations_dic[rel][relations_dic[rel]["s_idx"] != -1]

    relations_dic[rel]["o_idx"] = relations_dic[rel]["o"]
    relations_dic[rel]["o_idx"] = relations_dic[rel]["o_idx"].apply(
        lambda x: entites_dic[e2 + "_dic"][x] if x in entites_dic[e2 + "_dic"].keys() else -1)
    relations_dic[rel] = relations_dic[rel][relations_dic[rel]["o_idx"] != -1]

    relations_dic[rel] = relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
    rel_out = relations_dic[rel][["s_idx", "o_idx"]]
    if len(rel_out) > 0:
        map_folder = output_root_path + dataset_name + "/raw/relations/" + e1 + "___" + \
                     rel.split("/")[-1] + "___" + e2
        try:
            os.makedirs(map_folder, exist_ok=True)
        except:
            # pass
            print(f'^^^^^^^^^^^ Error exporting {rel_list[0]}')
        rel_out.to_csv(os.path.join(map_folder, "edge.csv.gz"), index=None, header=None,compression='gzip')
        ########## write relations num #################
        # with open(os.path.join(map_folder, "num-edge-list.csv.gz"), "w") as f:
        #     f.write(str(len(relations_dic[rel])))
        pd.DataFrame({0:[len(relations_dic[rel])]}).to_csv(os.path.join(map_folder, "num-edge-list.csv.gz"),header=None, index=None,compression='gzip')
        ##################### write relations idx #######################
        rel_idx = relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]
        rel_out["rel_idx"] = rel_idx
        rel_out["rel_idx"].to_csv(os.path.join(map_folder, "edge_reltype.csv.gz"), header=None, index=None,compression='gzip')
        return True
    else:
        return False

###################### Zip Folder to OGB Format
# zip -r mag_ComputerProgramming_papers_venue_QM3.zip mag_ComputerProgramming_papers_venue_QM3/ -i '*.gz'
def define_rel_types(g_tsv_df):
    g_tsv_df["p"]

def inference_transform_tsv_to_PYG(dataset_name, dataset_name_csv, dataset_types, target_rel, output_root_path
                                   , Header_row=None, targetNodeType=None):
    dic_results = {}  # args.dic_results #{}
    start_t = datetime.datetime.now()
    if dataset_types == "":
        dataset_types = output_root_path + dataset_name_csv + "_types.csv"

    if Header_row is None:
        g_tsv_df = pd.read_csv(output_root_path + dataset_name_csv + ".tsv", encoding_errors='ignore', sep="\t",header=Header_row)
        g_tsv_types_df = pd.read_csv(dataset_types, encoding_errors='ignore',header=None)
        g_tsv_df.columns=["s","p","o"]
        g_tsv_types_df.columns=['stype','ptype', 'otype']
    else:
        g_tsv_df = pd.read_csv(output_root_path + dataset_name_csv + ".tsv", encoding_errors='ignore', sep="\t",header=Header_row)
        g_tsv_types_df = pd.read_csv(dataset_types, encoding_errors='ignore',header=None)

    g_tsv_df.columns = ["s", "p", "o"]
    g_tsv_types_df.columns = ['stype', 'ptype', 'otype']
    #################### Remove '/' from the predicate of the Graph (tsv) ##############################
    g_tsv_df["p"] = g_tsv_df["p"].apply(lambda x: x.split('/')[-1].split('#')[-1])
    target_rel = target_rel.split('/')[-1].split('#')[-1]
    ##################Check if headers are present in _types file

    g_tsv_types_df.columns = ['stype', 'ptype', 'otype']

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
    relations_lst = g_tsv_df["p"].unique().astype("str").tolist()
    #relations_lst = [rel for rel in relations_lst if rel not in similar_target_rels]
    # print("relations_lst=", relations_lst)
    dic_results["usecase"] = dataset_name
    dic_results["TriplesCount"] = len(g_tsv_df)

    #################### Remove Split and Target Rel ############
    # if split_rel in relations_lst:
    #     relations_lst.remove(split_rel)
    if target_rel in relations_lst:
        relations_lst.remove(target_rel)
    # for srel in similar_target_rels:
    #     if srel in relations_lst:
    #         relations_lst.remove(srel)
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
    relations_df.to_csv(map_folder + "/relidx2relname.csv.gz", index=None,compression='gzip')
    # compress_gz(map_folder + "/relidx2relname.csv")
    ############################### create labels index ########################
    label_idx_df = pd.DataFrame(
        g_tsv_df[g_tsv_df["p"] == target_rel]["o"].apply(lambda x: str(x).strip()).unique().tolist(),
        columns=["label name"])
    dic_results["ClassesCount"] = len(label_idx_df)
    try:
        label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
    except:
        label_idx_df["label name"] = label_idx_df["label name"].astype("str")
        label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)

    label_idx_df["label idx"] = label_idx_df.index
    label_idx_df = label_idx_df[["label idx", "label name"]]
    label_idx_df.to_csv(map_folder + "/labelidx2labelname.csv.gz", index=None,compression='gzip')
    # compress_gz(map_folder + "/labelidx2labelname.csv")

    ###########################################prepare relations mapping#################################
    relations_entites_map = {}
    relations_dic = {}
    entites_dic = {}
    time_start_process_rln = datetime.datetime.now()
    """Orig process_relation"""
    for rel in relations_lst:
        try:
            rel_type = rel.split("/")[-1]
            rel_df = g_tsv_df[g_tsv_df["p"] == rel_type].reset_index(drop=True)
            rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([rel_type])]
            s_type = rel_types['stype'].values[0]
            o_type = rel_types['otype'].values[0]
            rel_df["s_type"] = s_type
            rel_df["o_type"] = o_type
            #########################################################################################
            rel_entity_types = rel_df[["s_type", "o_type"]].drop_duplicates()
            list_rel_types = []
            for idx, row in rel_entity_types.iterrows():
                list_rel_types.append((row["s_type"], rel, row["o_type"]))

            relations_entites_map[rel] = list_rel_types
            relations_dic[rel] = rel_df
            for rel_pair in list_rel_types:
                e1, rel, e2 = rel_pair
                if e1 != "literal" and e1 in entites_dic:
                    entites_dic[e1] = entites_dic[e1].union(
                        set(rel_df[rel_df["s_type"] == e1]["s"].unique()))  # .apply(
                elif e1 != "literal":
                    entites_dic[e1] = set(rel_df[rel_df["s_type"] == e1]["s"].unique())  # .apply(

                if e2 != "literal" and e2 in entites_dic:
                    entites_dic[e2] = entites_dic[e2].union(
                        set(rel_df[rel_df["o_type"] == e2]["o"].unique()))  # .apply(

                elif e2 != "literal":
                    entites_dic[e2] = set(rel_df[rel_df["o_type"] == e2]["o"].unique())  # .apply(

        except Exception as e:
            print(f"++++++++ DATA_TRANSFORMATION SKIPPING RELATION {rel}: {e} ++++++++")

    """ Parallel process_relation"""
    # manager = Manager()
    # entites_dic = manager.dict()
    # relations_entites_map = manager.dict()
    # relations_dic = manager.dict()
    #

    # with multiprocessing.Pool() as pool:
    #     pool.starmap(process_relation, [(rel, g_tsv_df, g_tsv_types_df, entites_dic, relations_entites_map, relations_dic) for rel in relations_lst])


    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_relation, rel, g_tsv_df, g_tsv_types_df, entites_dic, relations_entites_map, relations_dic)
    #                for rel in relations_lst]
        # for future in as_completed(futures):
        #     pass


    # print(f'Done, vars: emb_dict {entites_dic}\n relations_entites_map: {relations_entites_map}\n relations_dic:{relations_dic}')

    ############################### Obtain Target node if not explicitly given #########
    time_end_process_rln = (datetime.datetime.now() - time_start_process_rln).total_seconds()
    print (f'Process Relation Done! at {time_end_process_rln} s')
    if targetNodeType is None:
        # targetNodeType = list(g_tsv_types_df[g_tsv_types_df['ptype'] == target_rel]['stype'])[0]
        targetNodeType = \
        list(g_tsv_types_df[g_tsv_types_df['ptype'] == target_rel]['stype'].value_counts().to_dict().keys())[0]

    ############################### Make sure all target nodes have label ###########
    target_subjects_lst = g_tsv_df[g_tsv_df["p"] == target_rel]["s"].unique().tolist()  # .apply(
    entites_dic[targetNodeType] = set.intersection(entites_dic[targetNodeType], set(target_subjects_lst))



    ############################ write entites index #################################

    time_start_process_key = datetime.datetime.now()
    """ Orig process_key"""
    # for key in list(entites_dic.keys()):
    #     entites_dic[key] = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype(
    #         'str').sort_values(by="ent name").reset_index(drop=True)
    #     entites_dic[key] = entites_dic[key].drop_duplicates()
    #     entites_dic[key]["ent idx"] = entites_dic[key].index
    #     entites_dic[key] = entites_dic[key][["ent idx", "ent name"]]
    #     entites_dic[key + "_dic"] = pd.Series(entites_dic[key]["ent idx"].values,
    #                                           index=entites_dic[key]["ent name"]).to_dict()
    #     map_folder = output_root_path + dataset_name + "/mapping"
    #     try:
    #         os.stat(map_folder)
    #     except:
    #         os.makedirs(map_folder)
    #     entites_dic[key].to_csv(map_folder + "/" + key + "_entidx2name.csv", index=None)
    #     compress_gz(map_folder + "/" + key + "_entidx2name.csv")

    """ Parallel Process key"""
    manager = Manager()
    entites_dic = manager.dict(entites_dic)
    relations_dic = manager.dict(relations_dic)
    relations_entites_map = manager.dict(relations_entites_map)

    # entites_dic = dict(entites_dic)
    # relations_entites_map = dict(relations_entites_map)
    # relations_dic = dict(relations_dic)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_key, key, entites_dic, output_root_path, dataset_name)
                   for key in list(entites_dic.keys())]

        # Wait for all tasks to complete
        # for future in as_completed(futures):
        #     # Do nothing with the results, as they are updated in the shared dictionaries
        #     pass
    entites_dic = dict(entites_dic)
    relations_entites_map = dict(relations_entites_map)
    relations_dic = dict(relations_dic)
    #################### write nodes statistics ######################



    time_end_process_key = (datetime.datetime.now() - time_start_process_key).total_seconds()
    lst_node_has_feat = [
        list(
            filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
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
        index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-feat.csv")

    pd.DataFrame(lst_node_has_label).to_csv(
        output_root_path + dataset_name + "/raw/nodetype-has-label.csv.gz",
        header=None, index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/nodetype-has-label.csv")

    pd.DataFrame(lst_num_node_dict).to_csv(
        output_root_path + dataset_name + "/raw/num-node-dict.csv.gz", header=None,
        index=None,compression='gzip')
    # compress_gz(output_root_path + dataset_name + "/raw/num-node-dict.csv")

    ############################### create label relation index  ######################
    label_idx_df["label idx"] = label_idx_df["label idx"].astype("int64")
    label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
    ############ drop multiple targets per subject keep first#######################
    labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel].reset_index(drop=True)
    labels_rel_df = labels_rel_df.sort_values(['s', 'o'], ascending=[True, True])
    labels_rel_df = labels_rel_df.drop_duplicates(subset=["s"], keep='first')
    ###############################################################################
    rel_type = target_rel
    rel_types = g_tsv_types_df[g_tsv_types_df['ptype'].isin([rel_type])]
    # s_type = rel_types['stype'].values[0]
    o_type = rel_types['otype'].values[0]
    s_label_type = targetNodeType
    # o_label_type = o_type
    label_type = targetNodeType
    labels_rel_df["s_idx"] = labels_rel_df["s"]  # .apply(
    # lambda x: str(x).split("/")[-1])
    labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("str")
    # print("entites_dic=", list(entites_dic.keys()))
    labels_rel_df["s_idx"] = labels_rel_df["s_idx"].apply(
        lambda x: entites_dic[s_label_type + "_dic"][x] if x in entites_dic[
            s_label_type + "_dic"].keys() else -1)
    # labels_rel_df_notfound = labels_rel_df[labels_rel_df["s_idx"] == -1]
    labels_rel_df = labels_rel_df[labels_rel_df["s_idx"] != -1]
    labels_rel_df = labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)

    labels_rel_df["o_idx"] = labels_rel_df["o"]  # .apply(lambda x: str(x).split("/")[-1])
    labels_rel_df["o_idx"] = labels_rel_df["o_idx"].apply(
        lambda x: label_idx_dic[str(x)] if str(x) in label_idx_dic.keys() else -1)
    out_labels_df = labels_rel_df[["o_idx"]]
    map_folder = output_root_path + dataset_name + "/raw/node-label/" + s_label_type
    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)
    out_labels_df.to_csv(map_folder + "/node-label.csv.gz", header=None, index=None,compression='gzip')
    # compress_gz(map_folder + "/node-label.csv")
    ###########################################split parts (train/test/validate)#########################
    split_df = g_tsv_df[g_tsv_df["p"] == target_rel]

    ########## remove drug  with multi labels ################
    target_label_dict = split_df["s"].value_counts().to_dict()
    target_nodes_to_keep_lst = list(k for k, v in target_label_dict.items() if v == 1)
    split_df = split_df[split_df["s"].isin(target_nodes_to_keep_lst)]
    ########## remove labels with less than 9 samples################
    s_label_type = targetNodeType
    o_label_type = o_type
    label_type = s_label_type

    split_df["s"] = split_df["s"].astype("str").apply(lambda x: entites_dic[label_type + "_dic"][str(x)] if x in entites_dic[
        label_type + "_dic"] else -1)

    split_df = split_df[split_df["s"] != -1]
    label_type_values_lst = list(entites_dic[label_type + "_dic"].values())
    split_df = split_df[split_df["s"].isin(label_type_values_lst)]
    split_df = split_df.sort_values(by=["s"]).reset_index(drop=True)
    map_folder = output_root_path + dataset_name + "/split/" + label_type  # + split_by[        "folder_name"] + "/"

    try:
        os.stat(map_folder)
    except:
        os.makedirs(map_folder)

    split_df.to_csv(map_folder + "/test.csv.gz", index=None, header=None,compression='gzip')
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
    #             + "nodetype-has-split.csv")
    ###################### write entites relations for nodes only (non literals) #########################
    # entites_dic = dict(entites_dic)
    # relations_entites_map = dict(relations_entites_map)
    # relations_dic = dict(relations_dic)
    """ Orig Relation export"""
    time_start_export = datetime.datetime.now()
    # for rel_list in relations_entites_map.values():
    #     e1, rel, e2 = rel_list[0]
    #     ############
    #     relations_dic[rel]["s_idx"] = relations_dic[rel]["s"]  # .apply(
    #     relations_dic[rel]["s_idx"] = relations_dic[rel]["s_idx"].apply(
    #         lambda x: entites_dic[e1 + "_dic"][x] if x in entites_dic[
    #             e1 + "_dic"].keys() else -1)
    #     relations_dic[rel] = relations_dic[rel][relations_dic[rel]["s_idx"] != -1]
    #     ################
    #     relations_dic[rel]["o_idx"] = relations_dic[rel]["o"]  # .apply(
    #     # lambda x: str(x).split("/")[-1])
    #     relations_dic[rel]["o_idx"] = relations_dic[rel]["o_idx"].apply(
    #         lambda x: entites_dic[e2 + "_dic"][x] if x in entites_dic[
    #             e2 + "_dic"].keys() else -1)
    #     relations_dic[rel] = relations_dic[rel][relations_dic[rel]["o_idx"] != -1]
    #
    #     relations_dic[rel] = relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
    #     rel_out = relations_dic[rel][["s_idx", "o_idx"]]
    #     if len(rel_out) > 0:
    #         map_folder = output_root_path + dataset_name + "/raw/relations/" + e1 + "___" + \
    #                      rel.split("/")[-1] + "___" + e2
    #         try:
    #             os.stat(map_folder)
    #         except:
    #             os.makedirs(map_folder)
    #         rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
    #         compress_gz(map_folder + "/edge.csv")
    #         ########## write relations num #################
    #         f = open(map_folder + "/num-edge-list.csv", "w")
    #         f.write(str(len(relations_dic[rel])))
    #         f.close()
    #         compress_gz(map_folder + "/num-edge-list.csv")
    #         ##################### write relations idx #######################
    #         rel_idx = \
    #             relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]
    #         rel_out["rel_idx"] = rel_idx
    #         rel_idx_df = rel_out["rel_idx"]
    #         rel_idx_df.to_csv(map_folder + "/edge_reltype.csv", header=None, index=None)
    #         compress_gz(map_folder + "/edge_reltype.csv")
    #     else:
    #         lst_relations.remove([e1, str(rel).split("/")[-1], e2])
    #
    #     pd.DataFrame(lst_relations).to_csv(
    #         output_root_path + dataset_name + "/raw/triplet-type-list.csv",
    #         header=None, index=None)
    #     compress_gz(output_root_path + dataset_name + "/raw/triplet-type-list.csv")

            ###################Zip Folder ###############3
    """No!"""
        # shutil.make_archive(output_root_path + dataset_name, 'zip',
        #                     root_dir=output_root_path, base_dir=dataset_name)
    # entites_dic = dict(entites_dic)
    # relations_entites_map = dict(relations_entites_map)
    # relations_dic = dict(relations_dic)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(write_relations, rel_list, relations_dic, entites_dic, relations_df, output_root_path,
                            dataset_name)
            for rel_list in relations_entites_map.values()]

        # Wait for all tasks to complete
        non_empty_relation_lists = [future.result() for future in as_completed(futures)]

    # Remove empty relation lists
    non_empty_rel_lists = [rel_list for rel_list, non_empty in
                           zip(relations_entites_map.values(), non_empty_relation_lists) if non_empty]

    relations_entites_map = {idx: rel_list for idx, rel_list in enumerate(non_empty_rel_lists)}

    shutil.make_archive(output_root_path + dataset_name, 'zip',
                        root_dir=output_root_path, base_dir=dataset_name)
    time_end_export = (datetime.datetime.now() - time_start_export).total_seconds()
    end_t = datetime.datetime.now()
    dic_results['target_mapping'] = {v : k for k,v in entites_dic[targetNodeType + '_dic'].items()}
    dic_results["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()
    print("######## TRANSFORMATION COMPLETE ###############")
    print(f"######## Time Process Relation : {time_end_process_rln} #########")
    print(f"######## Time Process Key : {time_end_process_key} #########")
    print(f"######## Time Process Export : {time_end_export} #########")

    return dic_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSV to PYG')
    #parser.add_argument('--csv_path', type=str, default="")
    parser.add_argument('--target_rel', type=str, default="https://dblp.org/rdf/schema#publishedIn") # https://dblp.org/rdf/schema#publishedIn
    #split_data_type
    #size of train # if random then in % else specific value for train and valid split
    #size of valid
    parser.add_argument('--dataset_name',type=str, default="DBLP-Springer-Papers") # name of generated zip file
    #parser.add_argument('--dataset_csv',type=str, default="")
    parser.add_argument('--dataset_name_csv',type=str, default="DBLP-Springer-Papers")  # csv/tsv , input dataset name
    parser.add_argument('--dataset_types',type=str, default="") # path to the 'types' file containing relatiolns
    #parser.add_argument('--split_by', type=dict, default={}) replaced by train and valid args
    parser.add_argument('--split_rel', type=str, default="random") # https://dblp.org/rdf/schema#yearOfPublication #default = random # TODO could be null in some rows
    parser.add_argument('--split_rel_train_value', type=int, default=0)
    parser.add_argument('--split_rel_valid_value', type=int, default=0)
    parser.add_argument('--test_size',type=float,default=0.2)
    parser.add_argument('--valid_size',type=float,default=0.5)
    parser.add_argument('--similar_target_rels',type=list, default=[]) # 'https://dblp.org/rdf/schema#publishedInSeries' 
    # parser.add_argument('--target_node',type=str, default = "")
    #parser.add_argument('--dic_results',type=dict, default = "")
    parser.add_argument('--Literals2Nodes',type=bool, default = False) # convert literal vals into nodes (eg name of paper )or ignore 
    parser.add_argument('--output_root_path',type=str, default = "../../Datasets/")
    parser.add_argument('--MINIMUM_INSTANCE_THRESHOLD',type=int, default = 3)

    args = parser.parse_args()
    dataset_name = args.dataset_name  # "biokg_Drug_Classification" # Name of the dataset
    dataset_name_csv = args.dataset_name_csv  # "biokg"  # spo in IRI .csv no need for <>
    dataset_types = args.dataset_types  # "biokg_types.csv"  # kind of ontology
    split_rel = args.split_rel  # "http://purl.org/dc/terms/year"
    # split_by = args.split_by #{"folder_name": "random"}  # , "split_data_type": "int", "train":2006  ,"valid":2007 , "test":2008 }
    target_rel = args.target_rel  # "https://www.biokg.org/CLASS"  # is in the dataset and is StudiedDrug
    #similar_target_rels = args.similar_target_rels  # ["https://www.biokg.org/SUBCLASS", "https://www.biokg.org/SUPERCLASS"]
    # Literals2Nodes = args.Literals2Nodes  # False
    output_root_path = args.output_root_path  # "/home/ubuntu/flora_tests/biokg/data/"
    inference_transform_tsv_to_PYG(dataset_name,dataset_name_csv,dataset_types,target_rel,output_root_path,Header_row=True)



    
