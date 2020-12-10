import os
import sys
import argparse
import numpy as np 
import pandas as pd 
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='./data/', type=str)
parser.add_argument('--data_dir', default='./data/data.csv', type=str)
parser.add_argument('--n_splits', default='8', type=int)
parser.add_argument('--train_dir', default='./data/train.csv', type=str)
parser.add_argument('--val_dir', default='./data/val.csv', type=str)
parser.add_argument('--test_dir', default='./data/test.csv', type=str)

parser.add_argument('--random_state', default='2020', type=int)
args = parser.parse_args()
import csv
import random



if __name__ == '__main__':

    path_list = next(os.walk(args.datapath))[1]
    print(path_list)
    dataf = open(args.datapath+'train.csv','w',newline='')
    testf = open(args.datapath + 'test.csv', 'w', newline='')
    valf = open(args.datapath + 'val.csv', 'w', newline='')
    fcs=csv.writer(dataf)
    tfcs = csv.writer(testf)
    vfcs = csv.writer(valf)
    fcs.writerow(['filepath','target'])
    tfcs.writerow(['filepath', 'target'])
    vfcs.writerow(['filepath', 'target'])
    for pt in path_list:
        print(pt)
        name_list = os.listdir(args.datapath+pt)
        for nm in name_list:
            p=random.random()
            if p>0.3:
                fcs.writerow([os.path.join(args.datapath,pt,nm),pt])
            elif p>0.2:
                vfcs.writerow([os.path.join(args.datapath, pt, nm), pt])
            else:
                tfcs.writerow([os.path.join(args.datapath, pt, nm), pt])

    dataf.close()
    testf.close()
    valf.close()

    # df_data = pd.read_csv(args.data_dir)
    # img_path_list = df_data['filepath'].values.tolist()
    # label_list = df_data['target'].values.tolist()
    #
    #
    # data_label = []
    #
    #
    # for per_img_path, per_label in zip( img_path_list, label_list ):
    #     data_label.append( [ per_img_path, per_label ] )
    #
    # train_list = []
    # val_list = []
    # test_list = []
    # kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    # for index, (train_index, val_index) in enumerate(kf.split(data_label)):
    #     print(index)
    #     print(len(val_index))
    #     print(len(train_index))
    #     for i in val_index:
    #         data_label[i].append(index)
    #
    # data_label = np.array(data_label)
    # train_list = np.array( train_list )
    # val_list = np.array(val_list)
    # test_list = np.array(test_list)
    # # print (data_label)
    #
    # res = DataFrame()
    # res['filepath'] = data_label[:,0]
    # res['target'] = data_label[:,1]
    # res['fold'] = data_label[:,2]
    # res[ ['filepath', 'target', 'fold'] ].to_csv(args.train_dir, index=False)



    # res = DataFrame()
    # res['filepath'] = train_list[:,0]
    # res['target'] = train_list[:,1]
    # res['fold'] = train_list[:,2]
    # res[ ['filepath', 'target', 'fold'] ].to_csv(args.train_dir, index=False)
    # res = DataFrame()
    # res['filepath'] = val_list[:,0]
    # res['target'] = val_list[:,1]
    # res['fold'] = val_list[:,2]
    # res[ ['filepath', 'target', 'fold'] ].to_csv(args.val_dir, index=False)
    # res = DataFrame()
    # res['filepath'] = test_list[:,0]
    # res['target'] = test_list[:,1]
    # res['fold'] = test_list[:,2]
    # res[ ['filepath', 'target', 'fold'] ].to_csv(args.test_dir, index=False)


