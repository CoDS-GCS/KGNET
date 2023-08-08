import pickle
import pandas as pd

def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    rel_reidx = dict()
    relidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(rel_reidx), dict(ent_reidx)


def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        try:
            tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
        except Exception as e:
            print(e)
    return tri_reidx, dict(ent_reidx)


def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        try:
            tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
        except Exception as e:
            print(e)
    return tri_reidx


def data2pkl(data_name,target_rel='rel',BGP='FG'):
    train_tri = []
    file = open('../MetaEmbClean/data/{}/train.txt'.format(data_name))
    train_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    valid_tri = []
    file = open('../MetaEmbClean/data/{}/valid.txt'.format(data_name))
    valid_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    test_tri = []
    file = open('../MetaEmbClean/data/{}/test.txt'.format(data_name))
    test_tri.extend([l.strip().split() for l in file.readlines()])
    file.close()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    file = open('../MetaEmbClean/data/{}_ind/train.txt'.format(data_name))
    ind_train_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('../MetaEmbClean/data/{}_ind/valid.txt'.format(data_name))
    ind_valid_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    file = open('../MetaEmbClean/data/{}_ind/test.txt'.format(data_name))
    ind_test_tri = ([l.strip().split() for l in file.readlines()])
    file.close()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri}}

    pickle.dump(save_data, open(f'./data/{data_name}.pkl', 'wb'))

def data2pkl_Trans_to_Ind_BGP(data_name,BGP='FG',datapath='data',fsep='\t',logger=None):
    ind_train_size=0.2
    print("BGP=",BGP)
    # train_tri = []
    # train_f_name=''
    # test_f_name = ''
    # valid_f_name=''
    train_f_name = 'train.txt'
    valid_f_name = 'valid.txt'
    test_f_name = 'test.txt'
    # if BGP in['FG_ALL']:
    #     train_f_name='train.txt'
    #     valid_f_name='valid.txt'
    #     test_f_name='test.txt'
    # else:
    #     if BGP in['FG']:
    #         train_f_name='train.txt'
    #     else:
    #         train_f_name = 'train_'+target_rel+'_'+BGP+'.txt'
    #     valid_f_name ='valid_'+target_rel+'.txt'
    #     test_f_name = 'test_' + target_rel+'.txt'

    # print("train_f_name=",train_f_name)
    # print("test_f_name=",test_f_name)
    # print("valid_f_name=",valid_f_name)
    train_df = pd.read_csv(datapath+"/"+data_name+"/"+train_f_name, sep=fsep, header=None)
    train_df = train_df.rename(columns={0: "s", 1: "p", 2: "o"})
    train_df=train_df.apply(lambda x: x.str.strip())
    train_entites_lst = list(set(set(train_df["s"].tolist()).union(set(train_df["o"].tolist()))))
    # train_rel_lst = list(set(train_df["p"].tolist()))
    valid_df = pd.read_csv(datapath+"/"+data_name+"/"+valid_f_name, sep=fsep, header=None)
    valid_df = valid_df.rename(columns={0: "s", 1: "p", 2: "o"})
    valid_df=valid_df.apply(lambda x: x.str.strip())
    valid_entites_lst = list(set(set(valid_df["s"].tolist()).union(set(valid_df["o"].tolist()))))
    # valid_rel_lst = list(set(valid_df["p"].tolist()))
    test_df = pd.read_csv(datapath+"/"+data_name+"/"+test_f_name, sep=fsep, header=None)
    test_df = test_df.rename(columns={0: "s", 1: "p", 2: "o"})
    test_df=test_df.apply(lambda x: x.str.strip())
    test_entites_lst = list(set(set(test_df["s"].tolist()).union(set(test_df["o"].tolist()))))
    # test_rel_lst = list(set(test_df["p"].tolist()))

    df_train_not_test = train_df[(~train_df["s"].isin(test_entites_lst) & ~train_df["o"].isin(test_entites_lst))]
    df_test_g_train = train_df[(train_df["s"].isin(test_entites_lst) | train_df["o"].isin(test_entites_lst))]
    logger.info("df_test_g_train len="+str(len(df_test_g_train)))
    del train_df

    # Stratified Sampling
    # https://www.geeksforgeeks.org/stratified-sampling-in-pandas/
    df_train_g_test=df_train_not_test.groupby('p', group_keys=False).apply(lambda x: x.sample(frac=ind_train_size))
    # df_train_g_test = df_train_not_test.sample(frac=ind_train_size)
    df_train_g_train = df_train_not_test.drop(df_train_g_test.index)
    train_train_entites_lst = list(set(set(df_train_g_train["s"].tolist()).union(set(df_train_g_train["o"].tolist()))))
    df_missed_test_rows=df_train_g_test[(~df_train_g_test["s"].isin(train_train_entites_lst) | ~df_train_g_test["o"].isin(train_train_entites_lst))]
    df_train_g_test = df_train_g_test.drop(df_missed_test_rows.index)
    logger.info("df_train_g_test len=" + str(len(df_train_g_test)))
    df_train_g_train=pd.concat([df_train_g_train,df_missed_test_rows],ignore_index=True)
    logger.info("df_train_g_train len=" + str(len(df_train_g_train)))

    del df_missed_test_rows
    # print(len(valid_df))
    df_train_g_valid = valid_df[(~valid_df["s"].isin(test_entites_lst) & ~valid_df["o"].isin(test_entites_lst))]
    logger.info("df_train_g_valid len=" + str(len(df_train_g_valid)))
    df_test_g_valid = valid_df[(valid_df["s"].isin(test_entites_lst) | valid_df["o"].isin(test_entites_lst))]
    logger.info("df_test_g_valid len=" + str(len(df_test_g_valid)))
    del valid_df
    train_tri= df_train_g_train.values.tolist()
    valid_tri = df_train_g_valid.values.tolist()
    test_tri = df_train_g_test.values.tolist()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    ind_train_tri = df_test_g_train.values.tolist()
    ind_valid_tri = df_test_g_valid.values.tolist()
    ind_test_tri = test_df.values.tolist()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri= reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri}
                 }

    pickle.dump(save_data, open(datapath+'/'+data_name+'_'+BGP+'.pkl', 'wb'))

def data2pkl_Trans_to_Ind(datapath='data',data_name=None,fsep='\t',logger=None):
    if data_name==None:
        data_name=datapath.split("/")[-1]
    ind_train_size=0.2
    train_f_name = 'train.txt'
    valid_f_name = 'valid.txt'
    test_f_name = 'test.txt'
    train_df = pd.read_csv(datapath+data_name+"/"+train_f_name, sep=fsep, header=None)
    train_df.columns=["s","p","o"]
    train_df=train_df.apply(lambda x: x.str.strip())
    train_entites_lst = list(set(set(train_df["s"].tolist()).union(set(train_df["o"].tolist()))))
    # train_rel_lst = list(set(train_df["p"].tolist()))
    valid_df = pd.read_csv(datapath+data_name+"/"+valid_f_name, sep=fsep, header=None)
    valid_df.columns=["s","p","o"]
    valid_df=valid_df.apply(lambda x: x.str.strip())
    valid_entites_lst = list(set(set(valid_df["s"].tolist()).union(set(valid_df["o"].tolist()))))
    # valid_rel_lst = list(set(valid_df["p"].tolist()))
    test_df = pd.read_csv(datapath+data_name+"/"+test_f_name, sep=fsep, header=None)
    test_df.columns=["s","p","o"]
    test_df=test_df.apply(lambda x: x.str.strip())
    test_entites_lst = list(set(set(test_df["s"].tolist()).union(set(test_df["o"].tolist()))))
    # test_rel_lst = list(set(test_df["p"].tolist()))

    df_train_not_test = train_df[(~train_df["s"].isin(test_entites_lst) & ~train_df["o"].isin(test_entites_lst))]
    df_test_g_train = train_df[(train_df["s"].isin(test_entites_lst) | train_df["o"].isin(test_entites_lst))]
    logger.info("df_test_g_train len="+str(len(df_test_g_train)))
    del train_df

    # Stratified Sampling
    # https://www.geeksforgeeks.org/stratified-sampling-in-pandas/
    df_train_g_test=df_train_not_test.groupby('p', group_keys=False).apply(lambda x: x.sample(frac=ind_train_size))
    # df_train_g_test = df_train_not_test.sample(frac=ind_train_size)
    df_train_g_train = df_train_not_test.drop(df_train_g_test.index)
    train_train_entites_lst = list(set(set(df_train_g_train["s"].tolist()).union(set(df_train_g_train["o"].tolist()))))
    df_missed_test_rows=df_train_g_test[(~df_train_g_test["s"].isin(train_train_entites_lst) | ~df_train_g_test["o"].isin(train_train_entites_lst))]
    df_train_g_test = df_train_g_test.drop(df_missed_test_rows.index)
    logger.info("df_train_g_test len=" + str(len(df_train_g_test)))
    df_train_g_train=pd.concat([df_train_g_train,df_missed_test_rows],ignore_index=True)
    logger.info("df_train_g_train len=" + str(len(df_train_g_train)))

    del df_missed_test_rows
    # print(len(valid_df))
    df_train_g_valid = valid_df[(~valid_df["s"].isin(test_entites_lst) & ~valid_df["o"].isin(test_entites_lst))]
    logger.info("df_train_g_valid len=" + str(len(df_train_g_valid)))
    df_test_g_valid = valid_df[(valid_df["s"].isin(test_entites_lst) | valid_df["o"].isin(test_entites_lst))]
    logger.info("df_test_g_valid len=" + str(len(df_test_g_valid)))
    del valid_df
    train_tri= df_train_g_train.values.tolist()
    valid_tri = df_train_g_valid.values.tolist()
    test_tri = df_train_g_test.values.tolist()

    train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
    valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
    test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)

    ind_train_tri = df_test_g_train.values.tolist()
    ind_valid_tri = df_test_g_valid.values.tolist()
    ind_test_tri = test_df.values.tolist()

    test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
    test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
    test_test_tri= reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

    save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri},
                 'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri}
                 }

    pickle.dump(save_data, open(datapath+data_name+"/"+data_name+'.pkl', 'wb'))