import argparse
import pandas as pd

def generateIndDataset(datapath,fsep="\t"):
    train_df=pd.read_csv(datapath+"/train.txt",sep=fsep,header=None)
    train_df=train_df.rename(columns={0:"s",1:"p",2:"o"})
    train_entites_lst=list(set(set(train_df["s"].tolist()).union(set(train_df["o"].tolist()))))
    train_rel_lst = list(set(train_df["p"].tolist()))
    valid_df = pd.read_csv(datapath + "/valid_isConnectedTo.txt", sep=fsep, header=None)
    valid_df = valid_df.rename(columns={0: "s", 1: "p", 2: "o"})
    valid_entites_lst = list(set(set(valid_df["s"].tolist()).union(set(valid_df["o"].tolist()))))
    valid_rel_lst = list(set(valid_df["p"].tolist()))
    test_df = pd.read_csv(datapath + "/test_isConnectedTo.txt", sep=fsep, header=None)
    test_df = test_df.rename(columns={0: "s", 1: "p", 2: "o"})
    test_entites_lst = list(set(set(test_df["s"].tolist()).union(set(test_df["o"].tolist()))))
    test_rel_lst = list(set(test_df["p"].tolist()))

    # print(len(train_df))
    df_train_not_test = train_df[(~train_df["s"].isin(test_entites_lst) & ~train_df["o"].isin(test_entites_lst))]
    print(len(df_train_not_test))

    df_train_test=df_train_not_test.sample(frac=0.1)
    print(len(df_train_test))

    df_train_train=df_train_not_test.drop(df_train_test.index)
    print(len(df_train_train))

    df_train_test=train_df[(train_df["s"].isin(test_entites_lst) | train_df["o"].isin(test_entites_lst))]
    print(len(df_train_test))

    # print(len(valid_df))
    df_valid_not_test = valid_df[(~valid_df["s"].isin(test_entites_lst) & ~valid_df["o"].isin(test_entites_lst))]
    print(len(df_valid_not_test))
    df_valid_test = valid_df[(valid_df["s"].isin(test_entites_lst) | valid_df["o"].isin(test_entites_lst))]
    print(len(df_valid_test))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default='data/Yago3-10')
    args = parser.parse_args()
    generateIndDataset(args.dataPath)
