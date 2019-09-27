import os
import numpy as np
import random
import pickle
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir):
    """
    create a dataset description file, which consists of images, labels
    这包含了所有需要的数据了
    """
    dataset = dict()
    dataset['description'] = 'peta'
    dataset['root'] = '/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/dataset/peta/images/'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = range(35)
    # load PETA.MAT
    data = loadmat('/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/dataset/peta/PETA.mat')
    # print('xiaoyu:')
    # print(data)
    for idx in range(105):
        dataset['att_name'].append(data['peta'][0][0][1][idx, 0][0])

    for idx in range(19000):
        dataset['image'].append('%05d.png' % (idx+1))
        dataset['att'].append(data['peta'][0][0][0][idx, 4:].tolist())
    # print(type(dataset))
    with open(os.path.join(save_dir, 'peta_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)


def create_trainvaltest_split(traintest_split_file):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    这包含了将哪些图片设置为训练集、验证集和测试集
    """
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    partition['weight_train'] = []
    # load PETA.MAT
    data = loadmat("/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/dataset/peta/PETA.mat")
    for idx in range(5):
        train = (data['peta'][0][0][3][idx][0][0][0][0][:, 0]-1).tolist()
        val = (data['peta'][0][0][3][idx][0][0][0][1][:, 0]-1).tolist()
        test = (data['peta'][0][0][3][idx][0][0][0][2][:, 0]-1).tolist()
        trainval = train + val
        partition['train'].append(train)
        partition['val'].append(val)
        partition['trainval'].append(trainval)
        partition['test'].append(test)
        # weight
        weight_trainval = np.mean(data['peta'][0][0][0][trainval, 4:].astype('float32') == 1, axis=0).tolist()
        weight_train = np.mean(data['peta'][0][0][0][train, 4:].astype('float32') == 1, axis=0).tolist()
        # print("*"*100)
        # print(weight_train)
        # print(weight_trainval)
        # print(len(weight_train))
        # print(len(weight_trainval))
        # print(data['peta'][0][0][0][trainval, 4:].shape)
        # print(data['peta'][0][0][0][train, 4:].shape)
        partition['weight_trainval'].append(weight_trainval)
        partition['weight_train'].append(weight_train)
    # print('*'*100)
    # print(partition['weight_trainval'])
    # print(len(partition['weight_trainval']))
    with open(traintest_split_file, 'wb') as f:
        pickle.dump(partition, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="peta dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/dataset/peta/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="/home/thulab/Desktop/pedestrian-attribute-recognition-pytorch/dataset/peta/peta_partition.pkl")
    args = parser.parse_args()
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
