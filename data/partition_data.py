
import numpy as np
import torch
from torch.utils.data import Dataset

class Partition(Dataset):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
def partition_data(train_dataset, args, num_users):
    if args.non_iid == 0:
        user_groups = noniid_unequal(train_dataset, num_users)
    if args.non_iid == 1:
        user_groups = equal(train_dataset, num_users, args.num_class)
    if args.non_iid == 2:
        user_groups = Mix_up(train_dataset, num_users, args.uniform)
    if args.non_iid == 3:
        user_groups = Dirichlet(train_dataset, num_users, args.dir)
    if args.non_iid == 4:
        user_groups = IID(train_dataset, num_users)
    return user_groups

def Dirichlet(dataset, num_users, dir):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = np.array(dataset.targets)
    n_cls = (int(torch.max(torch.tensor(label)))) + 1
    n_data = data.shape[0]
    dir_level = dir

    cls_priors = np.random.dirichlet(alpha=[dir_level] * n_cls, size=num_users)
    # cls_priors_init = cls_priors # Used for verification
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    idx_list = [np.where(label == i)[0] for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    idx_worker = [[None] for i in range(num_users)]

    for curr_worker in range(num_users):
        for data_sample in range(n_data // num_users):
            curr_prior = prior_cumsum[curr_worker]
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            while cls_amount[cls_label] <= 0:
                # If you run out of samples
                correction = [[1 - cls_priors[i, cls_label]] * n_cls for i in range(num_users)]
                cls_priors = cls_priors / correction
                cls_priors[:, cls_label] = [0] * num_users
                curr_prior = np.cumsum(cls_priors, axis=1)
                cls_label = np.argmax(np.random.uniform() <= curr_prior)

            cls_amount[cls_label] -= 1
            if idx_worker[curr_worker] == [None]:
                idx_worker[curr_worker] = [idx_list[cls_label][0]]
            else:
                idx_worker[curr_worker] = idx_worker[curr_worker] + [idx_list[cls_label][0]]

            idx_list[cls_label] = idx_list[cls_label][1::]
    data_list = [idx_worker[curr_worker] for curr_worker in range(num_users)]
    return data_list
def Mix_up(dataset, num_users, uniform):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = dataset.targets
    n_workers = num_users
    homo_ratio = uniform
    
    n_data = data.shape[0]
    n_homo_data = int(n_data * homo_ratio)
    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo = np.array(range(0, n_homo_data))
        data_homo_list= np.split(data_homo, n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        # label_hetero_sorted, index = torch.sort(torch.tensor(label_hetero))
        idxs_labels = np.vstack((np.arange(n_homo_data, n_data), label_hetero))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        data_hetero_list = np.array(np.split(idxs, n_workers))
        data_hetero_list = np.random.permutation(data_hetero_list)
    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                        zip(data_homo_list, data_hetero_list)]

    return data_list
def IID(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def equal(dataset, num_users, num_class):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = num_users * num_class, int(dataset.data.shape[0] / num_users / num_class)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int)
    return dict_users
def noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = int(dataset.data.shape[0] / 50), 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 4
    max_shard = 20

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int)
    return dict_users