import pandas as pd
import gzip
import datetime
import os
import shutil
from sklearn.metrics import precision_recall_fscore_support as score


def compress_gz(f_path):
    f_in = open(f_path, 'rb')
    f_out = gzip.open(f_path + ".gz", 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()


###################### Zip Folder to OGB Format
# zip -r mag_ComputerProgramming_papers_venue_QM3.zip mag_ComputerProgramming_papers_venue_QM3/ -i '*.gz'

if __name__ == '__main__':
    csv_path = "/shared_mnt/DBLP/dblp-2022-03-01_URI_Only.tsv"
    split_rel = "https://dblp.org/rdf/schema#yearOfEvent"
    target_rel = "https://dblp.org/rdf/schema#publishedIn"
    label_node = "rec"
    # prefix_uri="https://dblp.org"
    label_type = "conf"
    classification_usecase = "conf"
    filter_publication_years = 2015
    affaliations_Coverage_df = pd.read_csv("/shared_mnt/DBLP/Sparql_Sampling_conf/BDLP_Papers_Per_Affaliation_conf.csv")
    affaliations_Coverage_df = affaliations_Coverage_df[affaliations_Coverage_df["do_train"] == 1].reset_index(
        drop=True)
    sampledQueries = {
        "StarQuery": "",
        "BStarQuery": "",
        "PathQuery": "",
        "BPathQuery": ""
    }
    dic_results = {}
    use_FM = True
    split_by = {"folder_name": "time", "split_data_type": "int", "train": 2019, "valid": 2020, "test": 2021}
    for i, aff_row in affaliations_Coverage_df.iterrows():
        if i >= 0:
            for sample_key in sampledQueries.keys():

                start_t = datetime.datetime.now()
                if use_FM == True:
                    dataset_name = "dblp-2022-03-01_URI_Only"
                    g_tsv_df = pd.read_csv("/shared_mnt/DBLP/" + dataset_name + ".tsv", sep="\t", header=None)
                    try:
                        g_tsv_df = g_tsv_df.rename(columns={0: "s", 1: "p", 2: "o"})
                    except:
                        print("g_tsv_df columns=", g_tsv_df.columns())
                else:
                    dataset_name = "OBGN_QM_DBLP_" + classification_usecase + "_" + sample_key + "Usecase_" + str(
                        int(aff_row["Q_idx"])) + "_" + str(
                        str(aff_row["affiliation"]).strip().replace(" ", "_").replace("/", "_").replace(",", "_"))
                    g_tsv_df = pd.read_csv("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + ".csv")
                    # print("g_tsv_df columns=",g_tsv_df)
                    try:
                        g_tsv_df = g_tsv_df.rename(columns={"subject": "s", "predicate": "p", "object": "o"})
                    except:
                        print("g_tsv_df columns=", g_tsv_df.columns())
                print("dataset_name=", dataset_name)
                # print("g_tsv_df columns=",g_tsv_df)
                dic_results[dataset_name] = {}
                dic_results[dataset_name]["q_idx"] = int(aff_row["Q_idx"])
                dic_results[dataset_name]["usecase"] = dataset_name
                dic_results[dataset_name]["sample_key"] = sample_key
                #################################Filter by years ######################
                yearOfPublication_df = g_tsv_df[g_tsv_df["p"] == "https://dblp.org/rdf/schema#yearOfPublication"]
                yearOfPublication_df["o"] = yearOfPublication_df["o"].astype("int64")
                remove_rec_years_lst = yearOfPublication_df[yearOfPublication_df["o"] < filter_publication_years][
                    "s"].unique().tolist()
                g_tsv_df = g_tsv_df[~g_tsv_df["s"].isin(remove_rec_years_lst)]
                ########################filter conf Only #############################
                rec_types = ['journals', 'conf', 'reference', 'books']
                rec_types.remove(classification_usecase)
                for rec_type in rec_types:
                    remove_recs = \
                    g_tsv_df[g_tsv_df["s"].str.startswith("https://dblp.org/" + label_node + "/" + rec_type)][
                        "s"].unique().tolist()
                    g_tsv_df = g_tsv_df[~g_tsv_df["s"].isin(remove_recs)]
                    g_tsv_df = g_tsv_df.dropna()
                    remove_recs = \
                    g_tsv_df[g_tsv_df["o"].str.startswith("https://dblp.org/" + label_node + "/" + rec_type)][
                        "o"].unique().tolist()
                    g_tsv_df = g_tsv_df[~g_tsv_df["o"].isin(remove_recs)]
                ########################delete non target papers #####################
                rec_rels = ["https://dblp.org/rdf/schema#yearOfPublication",
                            "https://dblp.org/rdf/rdf-schema#label",
                            "http://www.w3.org/2000/01/rdf-schema#label",
                            "https://dblp.org/rdf/schema#title",
                            "https://dblp.org/rdf/schema#primaryElectronicEdition",
                            "https://dblp.org/rdf/schema#orderedCreators",
                            "http://www.w3.org/2002/07/owl#sameAs",
                            "https://dblp.org/rdf/schema#publishedInBook",
                            "https://dblp.org/rdf/schema#publishedInJournalVolume",
                            "https://dblp.org/rdf/schema#publishedInJournalVolumeIssue",
                            "https://dblp.org/rdf/schema#pagination",
                            "https://dblp.org/rdf/schema#numberOfCreators",
                            "https://dblp.org/rdf/schema#publishedInJournal",
                            "https://dblp.org/rdf/schema#otherElectronicEdition",
                            "https://dblp.org/rdf/schema#publicationNote",
                            "https://dblp.org/rdf/schema#isbn",
                            "https://dblp.org/rdf/schema#thesisAcceptedBySchool",
                            "https://dblp.org/rdf/schema#archivedElectronicEdition",
                            "https://dblp.org/rdf/schema#publishedBy",
                            'https://dblp.org/rdf/schema#authoredBy',
                            "https://dblp.org/rdf/schema#monthOfPublication",
                            "https://dblp.org/rdf/schema#publishedInSeries",
                            "https://dblp.org/rdf/schema#publishedInSeriesVolume",
                            "https://dblp.org/rdf/schema#publishedInBookChapter",
                            'https://dblp.org/rdf/schema#doi',
                            'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                            'http://purl.org/spar/datacite/hasIdentifier',
                            'https://dblp.org/rdf/schema#listedOnTocPage',
                            'https://dblp.org/rdf/schema#publishedAsPartOf',
                            'https://dblp.org/rdf/schema#bibtexType',
                            'https://dblp.org/rdf/schema#editedBy',
                            'https://dblp.org/rdf/schema#wikidata',
                            split_rel]

                lst_targets = list(set(g_tsv_df[g_tsv_df["p"] == target_rel]["s"].tolist()))
                for rel in rec_rels:
                    rel_df = g_tsv_df[(g_tsv_df["p"] == rel) & (g_tsv_df["s"].str.startswith("https://dblp.org/rec/"))]
                    lst_rel_objects_old = rel_df["o"].unique().tolist()
                    to_delete_papers = rel_df[~rel_df["s"].isin(lst_targets)]["s"].unique().tolist()
                    g_tsv_df = g_tsv_df[~g_tsv_df["s"].isin(to_delete_papers)]
                    lst_rel_objects_new = \
                    g_tsv_df[(g_tsv_df["p"] == rel) & (g_tsv_df["s"].str.startswith("https://dblp.org/rec/"))][
                        "o"].unique().tolist()
                    lst_to_delete_objects = list(set(lst_rel_objects_old) - set(lst_rel_objects_new))
                    g_tsv_df = g_tsv_df[~g_tsv_df["s"].isin(lst_to_delete_objects)]

                lst_targets = list(set(g_tsv_df[g_tsv_df["p"] == target_rel]["s"].tolist()))
                lst_targets2 = list(set(g_tsv_df[g_tsv_df["s"].str.startswith("https://dblp.org/rec/")]["s"].tolist()))
                tergets_dff = list(set(lst_targets2).difference(lst_targets))
                g_tsv_df_diff = g_tsv_df[g_tsv_df["s"].isin(tergets_dff)]["p"].unique()
                ################################filter Objects ######################
                rel_df = g_tsv_df[g_tsv_df["o"].str.startswith("https://dblp.org/rec/")]
                to_delete_papers = rel_df[~rel_df["o"].isin(lst_targets)]["o"].unique().tolist()
                g_tsv_df = g_tsv_df[~g_tsv_df["o"].isin(to_delete_papers)]
                ################################# filter Authors ######################
                lst_papers_authors = g_tsv_df[g_tsv_df["p"] == "https://dblp.org/rdf/schema#authoredBy"][
                    "o"].unique().tolist()
                lst_all_authors = g_tsv_df[g_tsv_df["s"].str.startswith("https://dblp.org/pid/")]["s"].unique().tolist()
                lst_to_delete_authors = list(set(lst_all_authors) - set(lst_papers_authors))
                print("len lst_to_delete_authors=", len(lst_to_delete_authors))
                print("len(g_tsv_df) befors delete=", len(g_tsv_df))
                g_tsv_df = g_tsv_df[~g_tsv_df["s"].isin(lst_to_delete_authors)]
                print("len(g_tsv_df) after delete=", len(g_tsv_df))
                #####################################################################
                relations_lst = g_tsv_df["p"].unique().tolist()
                relations_lst.remove(split_rel)
                relations_lst.remove(target_rel)
                ################################write relations index ########################
                relations_df = pd.DataFrame(relations_lst, columns=["rel name"])
                relations_df["rel name"] = relations_df["rel name"].apply(lambda x: str(x).split("/")[-1])
                relations_df["rel idx"] = relations_df.index
                relations_df = relations_df[["rel idx", "rel name"]]
                map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/mapping"
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                relations_df.to_csv(map_folder + "/relidx2relname.csv", index=None)
                compress_gz(map_folder + "/relidx2relname.csv")
                ############################### create target label index ########################
                label_idx_df = pd.DataFrame(
                    g_tsv_df[g_tsv_df["p"] == target_rel]["o"].apply(lambda x: str(x).strip()).unique().tolist(),
                    columns=["label name"])
                try:
                    label_idx_df["label name"] = label_idx_df["label name"].astype("int64")
                    label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)
                except:
                    label_idx_df["label name"] = label_idx_df["label name"].astype("str")
                    label_idx_df = label_idx_df.sort_values(by=["label name"]).reset_index(drop=True)

                label_idx_df["label idx"] = label_idx_df.index
                label_idx_df = label_idx_df[["label idx", "label name"]]
                label_idx_df.to_csv(map_folder + "/labelidx2labelname.csv", index=None)
                compress_gz(map_folder + "/labelidx2labelname.csv")
                ###########################################prepare relations mapping#################################
                relations_entites_map = {}
                relations_dic = {}
                entites_dic = {}

                # literal_rels=["https://dblp.org/rdf/schema#doi",
                #               "https://dblp.org/rdf/schema#title",
                #               "https://dblp.org/rdf/schema#primaryElectronicEdition",
                #               "http://www.w3.org/2002/07/owl#sameAs",
                #               'https://dblp.org/rdf/schema#otherElectronicEdition',
                #               'https://dblp.org/rdf/schema#orcid',
                #               'https://dblp.org/rdf/schema#webpage',
                #               'https://dblp.org/rdf/schema#archivedElectronicEdition',
                #               'https://dblp.org/rdf/schema#otherHomepage',
                #               'https://dblp.org/rdf/schema#primaryHomepage',
                #               'https://dblp.org/rdf/schema#archivedWebpage',
                #               'https://dblp.org/rdf/schema#yearOfPublication']
                for rel in relations_lst:
                    rel_df = g_tsv_df[g_tsv_df["p"] == rel].reset_index(drop=True)
                    rel_df["s_type"] = rel_df["s"].apply(
                        lambda x: str(x).split("/")[3] if str(x).startswith("http") and len(str(x).split(
                            "/")) > 3 else "literal")  # get third parts from url i.e https://dblp.org/rec/journals/corr/abs-2111-03922 to be "rec"

                    rel_df = rel_df[~rel_df["s_type"].isin(["literal"])]
                    ##########################filter O types by uri pattern ################################
                    rel_df["o_type"] = rel_df["o"].apply(
                        lambda x: str(x).split("/")[3] if str(x).startswith("http") and len(
                            str(x).split("/")) > 3 else "literal")
                    if len(rel_df["o_type"].unique()) > 3:
                        if rel.startswith("http") and '#' in rel:  ## DOI special case
                            rel_df["o_type"] = rel.split("#")[-1]
                        else:
                            rel_df["o_type"] = "literal"

                    # if rel.startswith("http") and '#' in rel : ## DOI special case
                    #     # in literal_rels
                    #     rel_df["o_type"] = rel.split("#")[-1]
                    # else:
                    #     rel_df["o_type"]=rel_df["o"].apply(lambda x: str(x).split("/")[3] if str(x).startswith("http") and len(str(x).split("/"))>3 else "literal")
                    rel_df = rel_df[~rel_df["o_type"].isin(["literal"])]
                    rel_entity_types = rel_df[["s_type", "o_type"]].drop_duplicates()
                    list_rel_types = []
                    for idx, row in rel_entity_types.iterrows():
                        list_rel_types.append((row["s_type"], rel, row["o_type"]))

                    relations_entites_map[rel] = list_rel_types
                    if len(list_rel_types) > 2:
                        print(len(list_rel_types))
                    relations_dic[rel] = rel_df
                    # e1_list=list(set(relations_dic[rel]["s"].apply(lambda x:str(x).split("/")[:-1])))
                    for rel_pair in list_rel_types:
                        e1, rel, e2 = rel_pair
                        if e1 != "literal" and e1 in entites_dic:
                            entites_dic[e1] = entites_dic[e1].union(set(rel_df[rel_df["s_type"] == e1]["s"].apply(
                                lambda x: str(x).split("/" + e1 + "/")[-1]).unique()))
                        elif e1 != "literal":
                            entites_dic[e1] = set(rel_df[rel_df["s_type"] == e1]["s"].apply(
                                lambda x: str(x).split("/" + e1 + "/")[-1]).unique())

                        if e2 != "literal" and e2 in entites_dic:
                            entites_dic[e2] = entites_dic[e2].union(set(rel_df[rel_df["o_type"] == e2]["o"].apply(
                                lambda x: str(x).split("/" + e2 + "/")[-1]).unique()))
                        elif e2 != "literal":
                            entites_dic[e2] = set(rel_df[rel_df["o_type"] == e2]["o"].apply(
                                lambda x: str(x).split("/" + e2 + "/")[-1]).unique())

                ############################ write entites index #################################
                for key in list(entites_dic.keys()):
                    entites_dic[key] = pd.DataFrame(list(entites_dic[key]), columns=['ent name']).astype(
                        'str').sort_values(by="ent name").reset_index(drop=True)
                    entites_dic[key] = entites_dic[key].drop_duplicates()
                    entites_dic[key]["ent idx"] = entites_dic[key].index
                    entites_dic[key] = entites_dic[key][["ent idx", "ent name"]]
                    entites_dic[key + "_dic"] = pd.Series(entites_dic[key]["ent idx"].values,
                                                          index=entites_dic[key]["ent name"]).to_dict()
                    # print("key=",entites_dic[key+"_dic"])
                    map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/mapping"
                    try:
                        os.stat(map_folder)
                    except:
                        os.makedirs(map_folder)
                    entites_dic[key].to_csv(map_folder + "/" + key + "_entidx2name.csv", index=None)
                    compress_gz(map_folder + "/" + key + "_entidx2name.csv")
                #################### write nodes statistics ######################
                lst_node_has_feat = [
                    list(filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
                lst_node_has_label = lst_node_has_feat.copy()
                lst_num_node_dict = lst_node_has_feat.copy()
                lst_has_feat = []
                lst_has_label = []
                lst_num_node = []

                for entity in lst_node_has_feat[0]:
                    if str(entity) == str(label_node):
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

                map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw"
                print("map_folder=", map_folder)
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)

                pd.DataFrame(lst_node_has_feat).to_csv(
                    "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/nodetype-has-feat.csv", header=None,
                    index=None)
                compress_gz("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/nodetype-has-feat.csv")

                pd.DataFrame(lst_node_has_label).to_csv(
                    "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/nodetype-has-label.csv",
                    header=None, index=None)
                compress_gz("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/nodetype-has-label.csv")

                pd.DataFrame(lst_num_node_dict).to_csv(
                    "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/num-node-dict.csv", header=None,
                    index=None)
                compress_gz("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/num-node-dict.csv")

                ############################### create label relation index ######################
                label_idx_dic = pd.Series(label_idx_df["label idx"].values, index=label_idx_df["label name"]).to_dict()
                labels_rel_df = g_tsv_df[g_tsv_df["p"] == target_rel]
                label_type = labels_rel_df["s"].values[0]
                s_label_type = (label_type.split("/")[3] if label_type.startswith("http") and len(
                    label_type.split("/")) > 3 else "literal")
                # label_type = str(labels_rel_df["s"].values[0]).split("/")
                # label_type=label_type[len(label_type)-2]
                labels_rel_df["s_idx"] = labels_rel_df["s"].apply(lambda x: str(x).split("/" + s_label_type + "/")[-1])
                labels_rel_df["s_idx"] = labels_rel_df["s_idx"].astype("str")
                labels_rel_df["s_idx"] = labels_rel_df["s_idx"].apply(
                    lambda x: entites_dic[s_label_type + "_dic"][x] if x in entites_dic[
                        s_label_type + "_dic"].keys() else -1)
                labels_rel_df_notfound = labels_rel_df[labels_rel_df["s_idx"] == -1]
                labels_rel_df = labels_rel_df[labels_rel_df["s_idx"] != -1]
                labels_rel_df = labels_rel_df.sort_values(by=["s_idx"]).reset_index(drop=True)
                o_label_type = labels_rel_df["o"].values[0]
                o_label_type = (o_label_type.split("/")[3] if o_label_type.startswith("http") and len(
                    o_label_type.split("/")) > 3 else "literal")
                labels_rel_df["o_idx"] = labels_rel_df["o"].apply(lambda x: str(x).split("/" + o_label_type + "/")[-1])
                labels_rel_df["o_idx"] = labels_rel_df["o_idx"].apply(lambda x: label_idx_dic[x])
                out_labels_df = labels_rel_df[["o_idx"]]
                map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/node-label/" + s_label_type
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                out_labels_df.to_csv(map_folder + "/node-label.csv", header=None, index=None)
                compress_gz(map_folder + "/node-label.csv")
                ###########################################split parts (train/test/validate)#########################
                split_df = g_tsv_df[g_tsv_df["p"] == split_rel]
                label_type = str(split_df["s"].values[0])
                label_type = (label_type.split("/")[3] if label_type.startswith("http") and len(
                    label_type.split("/")) > 3 else "literal")
                try:
                    split_df["s"] = split_df["s"].apply(lambda x: str(x).split("/" + label_type + "/")[-1]).astype(
                        "int64").apply(
                        lambda x: entites_dic[label_type + "_dic"][x] if x in entites_dic[label_type + "_dic"] else -1)
                    # split_df = split_df[split_df["s"] != -1]
                except:
                    split_df["s"] = split_df["s"].apply(lambda x: str(x).split("/" + label_type + "/")[-1]).astype(
                        "str").apply(
                        lambda x: entites_dic[label_type + "_dic"][x] if x in entites_dic[label_type + "_dic"] else -1)

                split_df = split_df[split_df["s"] != -1]
                split_df["o"] = split_df["o"].astype(split_by["split_data_type"])
                label_type_values_lst = list(entites_dic[label_type + "_dic"].values())
                split_df = split_df[split_df["s"].isin(label_type_values_lst)]
                split_df = split_df.sort_values(by=["s"]).reset_index(drop=True)

                train_df = split_df[split_df["o"] <= split_by["train"]]["s"]
                valid_df = split_df[(split_df["o"] > split_by["train"]) & (split_df["o"] <= split_by["valid"])]["s"]
                test_df = split_df[(split_df["o"] > split_by["valid"]) & (split_df["o"] <= split_by["test"])]["s"]

                map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/split/" + split_by[
                    "folder_name"] + "/" + label_type
                try:
                    os.stat(map_folder)
                except:
                    os.makedirs(map_folder)
                train_df.to_csv(map_folder + "/train.csv", index=None, header=None)
                compress_gz(map_folder + "/train.csv")
                valid_df.to_csv(map_folder + "/valid.csv", index=None, header=None)
                compress_gz(map_folder + "/valid.csv")
                test_df.to_csv(map_folder + "/test.csv", index=None, header=None)
                compress_gz(map_folder + "/test.csv")
                ###################### create nodetype-has-split.csv#####################
                lst_node_has_split = [
                    list(filter(lambda entity: str(entity).endswith("_dic") == False, list(entites_dic.keys())))]
                lst_has_split = []
                for rel in lst_node_has_split[0]:
                    if rel == label_type:
                        lst_has_split.append("True")
                    else:
                        lst_has_split.append("False")
                lst_node_has_split.append(lst_has_split)
                pd.DataFrame(lst_node_has_split).to_csv(
                    "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/split/" + split_by[
                        "folder_name"] + "/nodetype-has-split.csv", header=None, index=None)
                compress_gz("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/split/" + split_by[
                    "folder_name"] + "/nodetype-has-split.csv")
                ############################ write entites relations  #################################
                idx = 0
                # for key in entites_dic["author_dic"].keys():
                #     print(key, entites_dic["author_dic"][key])
                #     idx=idx+1
                #     if idx>10:
                #         break;
                # print( list(entites_dic.keys()))
                for rel in relations_dic:
                    for rel_list in relations_entites_map[rel]:
                        e1, rel, e2 = rel_list
                        ############
                        relations_dic[rel]["s_idx"] = relations_dic[rel]["s"].apply(
                            lambda x: str(x).split("/" + e1 + "/")[-1])
                        relations_dic[rel]["s_idx"] = relations_dic[rel]["s_idx"].apply(
                            lambda x: entites_dic[e1 + "_dic"][x] if x in entites_dic[e1 + "_dic"].keys() else -1)
                        relations_dic[rel] = relations_dic[rel][relations_dic[rel]["s_idx"] != -1]
                        ################
                        # relations_dic[rel]["o_keys"]=relations_dic[rel]["o"].apply(lambda x:x.split("/")[3] if x.startswith("http") and len(x.split("/")) > 3 else x)
                        relations_dic[rel]["o_idx"] = relations_dic[rel]["o"].apply(
                            lambda x: str(x).split("/" + e2 + "/")[-1])
                        relations_dic[rel]["o_idx"] = relations_dic[rel]["o_idx"].apply(
                            lambda x: entites_dic[e2 + "_dic"][x] if x in entites_dic[e2 + "_dic"].keys() else -1)
                        relations_dic[rel] = relations_dic[rel][relations_dic[rel]["o_idx"] != -1]

                        relations_dic[rel] = relations_dic[rel].sort_values(by="s_idx").reset_index(drop=True)
                        rel_out = relations_dic[rel][["s_idx", "o_idx"]]
                        if len(rel_out) > 0:
                            map_folder = "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/relations/" + e1 + "___" + \
                                         rel.split("/")[-1] + "___" + e2
                            try:
                                os.stat(map_folder)
                            except:
                                os.makedirs(map_folder)
                            rel_out.to_csv(map_folder + "/edge.csv", index=None, header=None)
                            compress_gz(map_folder + "/edge.csv")
                            ########## write relations num #################
                            f = open(map_folder + "/num-edge-list.csv", "w")
                            f.write(str(len(relations_dic[rel])))
                            f.close()
                            compress_gz(map_folder + "/num-edge-list.csv")
                            ##################### write relations idx #######################
                            rel_idx = relations_df[relations_df["rel name"] == rel.split("/")[-1]]["rel idx"].values[0]
                            rel_out["rel_idx"] = rel_idx
                            rel_idx_df = rel_out["rel_idx"]
                            rel_idx_df.to_csv(map_folder + "/edge_reltype.csv", header=None, index=None)
                            compress_gz(map_folder + "/edge_reltype.csv")
                        else:
                            lst_relations.remove([e1, str(rel).split("/")[-1], e2])

                    pd.DataFrame(lst_relations).to_csv(
                        "/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/triplet-type-list.csv",
                        header=None, index=None)
                    compress_gz("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name + "/raw/triplet-type-list.csv")
                    #####################Zip Folder ###############3
                shutil.make_archive("/shared_mnt/DBLP/Sparql_Sampling_conf/" + dataset_name, 'zip',
                                    root_dir="/shared_mnt/DBLP/Sparql_Sampling_conf/", base_dir=dataset_name)
                # shutil.rmtree("/shared_mnt/DBLP/Sparql_Sampling_conf/"+dataset_name)
                end_t = datetime.datetime.now()
                print("DBLP_QM_conf_csv_to_Hetrog_time=", end_t - start_t, " sec.")
                if use_FM == True:
                    dic_results[dataset_name]["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()
                    pd.DataFrame(dic_results).transpose().to_csv(
                        "/shared_mnt/DBLP/Sparql_Sampling_conf/OGBN_BDLP_conf_FM_Uscases_CSVtoHetrog_times" + ".csv",
                        index=False)
                    break;
                # print(entites_dic)
                else:
                    dic_results[dataset_name]["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()
                    pd.DataFrame(dic_results).transpose().to_csv(
                        "/shared_mnt/DBLP/Sparql_Sampling_conf/OGBN_BDLP_conf_QM_Uscases_CSVtoHetrog_times" + ".csv",
                        index=False)
                # print(entites_dic)
            if use_FM == True:
                break;
