import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import pandas as pd
from utils import load_data, generate_sampled_graph_and_labels, build_test_graph, calc_mrr
from models import RGCN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate LP BGPs')
    parser.add_argument("--data_path", type=str, default="/media/hussein/UbuntuData/GithubRepos/RGCN/data/")
    parser.add_argument("--dataset", type=str, default="wikikg-v2-2015")
    parser.add_argument("--target_edge", type=str, default="P57")  # direector of
    # parser.add_argument("--dataset", type=str, default="FB15k-237")
    # parser.add_argument("--target_edge", type=str, default="/people/person/profession")
    # parser.add_argument("--dataset", type=str, default="wn18")
    # parser.add_argument("--target_edge", type=str, default="_hyponym")
    # parser.add_argument("--dataset", type=str, default="FB15K-273")
    # parser.add_argument("--target_edge", type=str, default="/people/person/profession")
    # parser.add_argument("--dataset", type=str, default="Yago10")
    # # parser.add_argument("--target_edge", type=str, default="isAffiliatedTo")
    # parser.add_argument("--target_edge", type=str, default="isLocatedIn")

    args = parser.parse_args()

    # valid_ds=pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/valid.txt", dtype=str,sep="\t", header=None)
    # valid_ds=valid_ds.rename(columns={0:'s',1:'p',2:'o'})
    # print(valid_ds["p"].value_counts())
    # valid_ds=valid_ds[valid_ds["p"].isin([selected_edge])]
    # print(valid_ds["p"].value_counts())
    # print(valid_ds)
    # valid_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/valid_"+selected_edge+".txt",header=None,index=None, sep="\t")
    # ###########################
    # test_ds = pd.read_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/test.txt",dtype=str, sep="\t", header=None)
    # test_ds = test_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # print(test_ds["p"].value_counts())
    # test_ds = test_ds[test_ds["p"].isin([selected_edge])]
    # print(test_ds["p"].value_counts())
    # print(test_ds)
    # test_ds.to_csv("/media/hussein/UbuntuData/GithubRepos/RGCN/data/wikikg-v2-2015/test_"+selected_edge+".txt", header=None,
    #                 index=None, sep="\t")
    target_edge = args.target_edge
    target_edge_name=target_edge.split("/")[-1]
    data_path = args.data_path+args.dataset+"/"
    ############################ read train-test-valid#####################
    file_sep=","
    train_ds = pd.read_csv(data_path + "train.txt", dtype=str, sep=file_sep, header=None)
    train_ds.to_csv(data_path + "train_t.txt", sep="\t", header=None,index=None)
    train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    print(len(train_ds[train_ds["p"].isin([target_edge])]["o"].unique().tolist()))
    print(train_ds)
    print(train_ds["p"].value_counts())
    lst_rels = train_ds["p"].unique().tolist()
    # for rel in lst_rels:
    #     s_counts = len(train_ds[train_ds["p"].isin([rel])]["s"].unique().tolist())
    #     o_counts = len(train_ds[train_ds["p"].isin([rel])]["o"].unique().tolist())
    #     print(rel, len(train_ds[train_ds["p"].isin([rel])]), s_counts, o_counts, o_counts / s_counts)

    valid_ds = pd.read_csv(data_path + "valid.txt", dtype=str, sep=file_sep, header=None)
    valid_ds.to_csv(data_path + "valid_t.txt", sep="\t", header=None,index=None)
    valid_ds = valid_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    print(valid_ds)
    test_ds = pd.read_csv(data_path + "test.txt", dtype=str, sep=file_sep, header=None)
    test_ds.to_csv(data_path + "test_ds_t.txt", sep="\t", header=None,index=None)
    test_ds = test_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    print(test_ds)


    # #########################Generate Dic ######################
    # all_triples_df=pd.concat([train_ds,valid_ds,test_ds])
    # relations_lst=all_triples_df["p"].unique().tolist()
    # relations_lst.sort()
    # pd.DataFrame(relations_lst).to_csv(data_path+"relations.dict",sep="\t",header=None)
    # entities_lst=list(set(all_triples_df["s"].unique().tolist()+ all_triples_df["o"].unique().tolist()))
    # entities_lst.sort()
    # pd.DataFrame(entities_lst).to_csv(data_path + "entities.dict", sep="\t", header=None)
    # #########################Generate valid_test target ######################
    # valid_ds[valid_ds["p"].isin([target_edge])].to_csv(data_path + "valid_"+target_edge_name+".txt", sep="\t", header=None,index=None)
    # test_ds[test_ds["p"].isin([target_edge])].to_csv(data_path + "test_" + target_edge_name + ".txt", sep="\t",header=None, index=None)
    # #########################Generate BGPS ######################
    # source_en = train_ds[train_ds["p"].isin([target_edge])]["s"].unique().tolist()
    # des_en = train_ds[train_ds["p"].isin([target_edge])]["o"].unique().tolist()
    # #########################################
    # train_SQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["s"].isin(des_en)))]
    # train_SQ = train_SQ.drop_duplicates()
    # print(train_SQ)
    # train_SQ.to_csv(data_path + "train_" + target_edge_name + "_SQ.txt", header=None, index=None, sep="\t")
    # ######################################
    # train_BSQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(source_en))
    #                       | (train_ds["s"].isin(des_en)) | (train_ds["o"].isin(des_en)))]
    # train_BSQ = train_BSQ.drop_duplicates()
    # print(train_BSQ)
    # train_BSQ.to_csv(data_path + "train_" + target_edge_name + "_BSQ.txt", header=None, index=None, sep="\t")
    # # #################################
    # source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    # des_des_en = train_ds[train_ds["s"].isin(des_en)]["o"].unique().tolist()
    # train_SQ_SQ = train_ds[(train_ds["s"].isin(source_source_en) | train_ds["s"].isin(des_des_en))]
    # train_PQ = pd.concat([train_SQ, train_SQ_SQ])
    # train_PQ = train_PQ.drop_duplicates()
    # print(train_PQ)
    # train_PQ.to_csv(data_path + "train_" + target_edge_name + "_PQ.txt", header=None, index=None, sep="\t")
    # #################################
    # train_q1 = train_ds[train_ds["s"].isin(source_en) | train_ds["o"].isin(source_en)]
    # train_q2 = train_ds[train_ds["s"].isin(des_en) | train_ds["o"].isin(des_en)]
    # train_q3 = train_ds[train_ds["s"].isin(source_source_en) | train_ds["o"].isin(source_source_en)]
    # train_q4 = train_ds[train_ds["s"].isin(des_des_en) | train_ds["o"].isin(des_des_en)]
    # train_BPQ = pd.concat([train_q1, train_q2, train_q3, train_q4])
    # train_BPQ = train_BPQ.drop_duplicates()
    # print(train_BPQ)
    # train_BPQ.to_csv(data_path + "train_" + target_edge_name + "_BPQ.txt", header=None, index=None, sep="\t")