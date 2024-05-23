import pandas as pd
import datetime
import requests
import traceback
import sys
import argparse
from threading import Thread
import threading
import os
import shutil
from Constants import *

from sklearn.model_selection import train_test_split
def transform_LP_train_valid_test_subsets(data_path,ds_name,target_rel,valid_size=0.1,test_size=0.1,delm='\t',containHeader=False,split_rel=None,inference=False):
    start_t = datetime.datetime.now()
    dic_results={}
    if containHeader:
        full_ds = pd.read_csv(os.path.join(data_path, ds_name + ".tsv"), dtype=str, sep=delm)
    else:
        full_ds= pd.read_csv(os.path.join(data_path, ds_name+".tsv"), dtype=str, sep=delm, header=None)
    full_ds.columns=["s","p","o"]
    dic_results["TriplesCount"] = len(full_ds)
    ##############################
    path = os.path.join(data_path, ds_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    print("Directory '% s' created" % path)
    #############################
    relations_lst = full_ds["p"].unique().tolist()
    relations_lst.sort()
    pd.DataFrame(relations_lst).to_csv(path + "/relations.dict", sep="\t", header=None)
    entities_lst = list(set(full_ds["s"].unique().tolist() + full_ds["o"].unique().tolist()))
    entities_lst=[str(elem) for elem in entities_lst]
    entities_lst.sort()
    pd.DataFrame(entities_lst).to_csv(path + "/entities.dict", sep="\t", header=None)

    split_df = full_ds[full_ds["p"].isin([target_rel])].reset_index(drop=True)
    if not inference:
        full_ds = full_ds[~full_ds["p"].isin([target_rel])]
        dic_results["testSize"] = test_size
        dic_results["validSize"] = valid_size
        dic_results["trainSize"] = 1 - (valid_size + test_size)

        test_size = test_size + valid_size
        if split_rel is None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(split_df.index.tolist(), split_df["o"].tolist(),test_size=test_size, random_state=42,stratify=split_df["o"].tolist())
            except:
                X_train, X_test, y_train, y_test = train_test_split(split_df.index.tolist(), split_df["o"].tolist(),test_size=test_size, random_state=42)

            test_size = test_size/(test_size + valid_size)
            try:
                X_valid,X_test, y_valid, y_test = train_test_split(X_test,y_test,test_size=test_size, random_state=42,stratify=X_test["o"].tolist())
            except:
                X_valid,X_test, y_valid, y_test = train_test_split(X_test, y_test,test_size=test_size, random_state=42)
        else:
            dic_results["splitEdge"] =split_rel

        split_df.iloc[X_valid].to_csv(path + "/valid.txt", sep="\t", header=None, index=None)
        split_df.iloc[X_test].to_csv(path + "/test.txt", sep="\t", header=None, index=None)

        X_train = pd.concat([full_ds, split_df.iloc[X_train]])
        X_train.to_csv(path + "/train.txt", sep="\t", header=None, index=None)
        end_t = datetime.datetime.now()
        dic_results["csv_to_Hetrog_time"] = (end_t - start_t).total_seconds()
        return dic_results

    else:
        full_ds.to_csv(path + "/train.txt", sep="\t", header=None, index=None)
        split_df.to_csv(path + "/test.txt", sep="\t", header=None, index=None)
        split_df.to_csv(path + "/valid.txt", sep="\t", header=None, index=None)
        return

###############################################################################################################################################
def write_entites_rels_dict(train_ds, valid_ds, test_ds,data_path):
    all_triples_df = pd.concat([train_ds, valid_ds, test_ds])
    relations_lst = all_triples_df["p"].unique().tolist()
    relations_lst.sort()
    pd.DataFrame(relations_lst).to_csv(data_path + "relations.dict", sep="\t", header=None)
    entities_lst = list(set(all_triples_df["s"].unique().tolist() + all_triples_df["o"].unique().tolist()))
    del all_triples_df
    entities_lst.sort()
    pd.DataFrame(entities_lst).to_csv(data_path + "entities.dict", sep="\t", header=None)
def write_valid_test_targetrel_subsets(target_rel_uri,target_rel_name,valid_ds,test_ds,delm='\t'):
    valid_ds[valid_ds["p"].isin([target_rel_uri])].to_csv(data_path + "valid_" + target_rel_name + ".txt", sep=delm,
                                                          header=None, index=None)
    test_ds[test_ds["p"].isin([target_rel_uri])].to_csv(data_path + "test_" + target_rel_name + ".txt", sep=delm,
                                                        header=None, index=None)
def write_d1h1_TOSG(train_ds,source_en,des_en,delm='\t'):
    train_SQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["s"].isin(des_en)))]
    train_SQ = train_SQ.drop_duplicates()
    print("len d1h1 =", len(train_SQ))
    train_SQ.to_csv(data_path + "train_" + target_rel_name + "_d1h1.txt", header=None, index=None, sep=delm)
def write_d2h1_TOSG(train_ds,source_en,des_en,delm='\t'):
    train_BSQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["o"].isin(source_en))
                          | (train_ds["s"].isin(des_en)) | (train_ds["o"].isin(des_en)))]
    train_BSQ = train_BSQ.drop_duplicates()
    # print(train_BSQ)
    print("len d2h1 =", len(train_BSQ))
    train_BSQ.to_csv(data_path + "train_" + target_rel_name + "_d2h1.txt", header=None, index=None, sep=delm)
def write_d1h2_TOSG(train_ds,source_en,des_en,delm='\t'):
    train_SQ = train_ds[((train_ds["s"].isin(source_en)) | (train_ds["s"].isin(des_en)))]
    train_SQ = train_SQ.drop_duplicates()
    source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    des_des_en = train_ds[train_ds["s"].isin(des_en)]["o"].unique().tolist()
    train_SQ_SQ = train_ds[(train_ds["s"].isin(source_source_en) | train_ds["s"].isin(des_des_en))]
    train_PQ = pd.concat([train_SQ, train_SQ_SQ])
    train_PQ = train_PQ.drop_duplicates()
    # print(train_PQ)
    print("len d1h2 =", len(train_PQ))
    train_PQ.to_csv(data_path + "train_" + target_rel_name + "_d1h2.txt", header=None, index=None, sep=delm)
def write_d2h2_TOSG(train_ds,source_en,des_en,delm='\t'):
    train_q1 = train_ds[train_ds["s"].isin(source_en) | train_ds["o"].isin(source_en)]
    train_q2 = train_ds[train_ds["s"].isin(des_en) | train_ds["o"].isin(des_en)]
    source_source_en = train_ds[train_ds["o"].isin(source_en)]["s"].unique().tolist()
    des_des_en = train_ds[train_ds["s"].isin(des_en)]["o"].unique().tolist()
    train_q3 = train_ds[train_ds["s"].isin(source_source_en) | train_ds["o"].isin(source_source_en)]
    train_q4 = train_ds[train_ds["s"].isin(des_des_en) | train_ds["o"].isin(des_des_en)]
    train_BPQ = pd.concat([train_q1, train_q2, train_q3, train_q4])
    train_BPQ = train_BPQ.drop_duplicates()
    # print(train_BPQ)
    print("len d2h2 =", len(train_BPQ))
    train_BPQ.to_csv(data_path + "train_" + target_rel_name + "_d2h2.txt", header=None, index=None, sep=delm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #parser.add_argument('--start_offset', dest='start_offset', type=int, help='Add start_offset', default=0)
    #parser.add_argument('--sparql_endpoint', type=str, help='SPARQL endpoint URL', default='http://206.12.98.118:8890/sparql')
    #parser.add_argument('--graph_uri', type=str, help=' KG URI', default='http://dblp.org')
    parser.add_argument('--target_rel_uri', type=str, help='target_rel_uri URI',default='http://www.wikidata.org/entity/P101')
    parser.add_argument("--data_path", type=str, default=KGNET_Config.datasets_output_path)
    parser.add_argument("--dataset", type=str, default="mid-ddc400fac86bd520148e574f86556ecd19a9fb9ce8c18ce3ce48d274ebab3965")
    parser.add_argument('--TOSG', type=str, help='TOSG Pattern',default='d1h1')
    parser.add_argument('--file_sep', type=str, help='triple delimter', default='\t')
    # parser.add_argument('--batch_size', type=int, help='batch_size', default='1000000')
    # parser.add_argument('--out_file', dest='out_file', type=str, help='output file to write trplies to', default='dblp_pv.tsv')
    # parser.add_argument('--threads_count', dest='threads_count', type=int, help='output file to write trplies to', default=64)

    args = parser.parse_args()
    print('args=',args)
    # start_offset = args.start_offset
    # graph_uri=args.graph_uri
    # sparql_endpoint =args.sparql_endpoint
    # batch_size=args.batch_size
    target_rel_name=args.target_rel_uri.split("/")[-1]
    target_rel_uri = args.target_rel_uri
    TOSG=args.TOSG
    data_path = args.data_path + args.dataset + "/"
    dataset = args.dataset
    dic_sep={'comma':',','tab':'\t'}
    file_sep = dic_sep[args.file_sep] if args.file_sep in dic_sep.keys() else args.file_sep
    ############################ convert TSV to pyg dataset ########################
    data_path = args.data_path
    ds_name = dataset
    transform_LP_train_valid_test_subsets(data_path, ds_name, args.target_rel_uri, valid_size=0.1, test_size=0.1, delm='\t',
                                          containHeader=False, split_rel=None)
    ############################ read train-test-valid daatsets #####################
    train_ds = pd.read_csv(data_path + "train.txt", dtype=str, sep=file_sep, header=None)
    train_ds = train_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    print(len(train_ds[train_ds["p"].isin([target_rel_uri])]["o"].unique().tolist()))
    # print(train_ds)
    # print(train_ds["p"].value_counts())
    lst_rels = train_ds["p"].unique().tolist()
    valid_ds = pd.read_csv(data_path + "valid.txt", dtype=str, sep=file_sep, header=None)
    # valid_ds.to_csv(data_path + "valid_t.txt", sep="\t", header=None,index=None)
    valid_ds = valid_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # print(valid_ds)
    test_ds = pd.read_csv(data_path + "test.txt", dtype=str, sep=file_sep, header=None)
    # test_ds.to_csv(data_path + "test_ds_t.txt", sep="\t", header=None,index=None)
    test_ds = test_ds.rename(columns={0: 's', 1: 'p', 2: 'o'})
    # print(test_ds)
    #####################################
    write_entites_rels_dict(train_ds, valid_ds, test_ds, data_path)
    write_valid_test_targetrel_subsets(target_rel_uri,target_rel_name,valid_ds,test_ds,delm='\t')
    del valid_ds
    del test_ds
    #########################Generate TOSG ######################
    start_t = datetime.datetime.now()
    source_en = train_ds[train_ds["p"].isin([target_rel_uri])]["s"].unique().tolist()
    des_en = train_ds[train_ds["p"].isin([target_rel_uri])]["o"].unique().tolist()
    #########################################
    q_start_t = datetime.datetime.now()
    if TOSG=='d1h1':
        write_d1h1_TOSG(train_ds,source_en,des_en,delm='\t')
    elif TOSG=='d1h2':
        write_d1h2_TOSG(train_ds,source_en,des_en,delm='\t')
    elif TOSG=='d2h1':
        write_d2h1_TOSG(train_ds,source_en,des_en,delm='\t')
    elif TOSG=='d2h2':
        write_d2h2_TOSG(train_ds,source_en,des_en,delm='\t')
    q_end_t = datetime.datetime.now()
    print("total time ", q_end_t - q_start_t, " sec.")