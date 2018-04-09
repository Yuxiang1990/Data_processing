from collections.abc import Sequence
import random

import numpy as np
import pandas as pd
import _init_path
from mylib.myclass.path_manager import INVASION_PATH
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple
print(INVASION_PATH.annotation)
INFO = pd.read_csv(INVASION_PATH.annotation)

LABEL = ['AAH', 'AIS', 'MIA', 'IA', 'OTHER']
# LABEL = ['AAH', 'AIS', 'MIA', 'IA']
LABEL_NAME = 'invasion_label'
invasion = "subset"
maglinant = "maglinant_subset"


class ClfDataset(Sequence):
    def __init__(self, crop_size=32, move=3, subset=[0, 1, 2, 3], define_label=lambda l: l):
        index = []
        for sset in subset:
            index += list(INFO[INFO[invasion] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = np.array([[label == s for label in LABEL] for s in INFO.loc[self.index, LABEL_NAME]])
        self.transform = Transform(crop_size, move)
        self.define_label = define_label

    def __getitem__(self, item):
        path = INFO.loc[self.index[item], 'path']
        # print(path)
        with np.load(path) as npz:
            voxel = self.transform(npz['raw'])
        label = self.label[item]
        return voxel, self.define_label(label)

    def __len__(self):
        return len(self.index)

    @classmethod
    def get_loader(cls, batch_size, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        total_size = len(dataset)
        print('Size', total_size)
        index_generator = shuffle_iterator(range(total_size))
        while True:
            data = []
            for _ in range(batch_size):
                idx = next(index_generator)
                data.append(dataset[idx])
            yield dataset._collate_fn(data)

    @classmethod
    def get_balanced_loader(cls, batch_sizes, *args, **kwargs):
        assert len(batch_sizes) == len(LABEL)
        dataset = cls(*args, **kwargs)
        total_size = len(dataset)
        print('Size', total_size)
        index_generators = []
        for l_idx in range(len(batch_sizes)):
            # this must be list, or `l_idx` will not be eval
            iterator = [i for i in range(total_size) if dataset.label[i, l_idx]]
            index_generators.append(shuffle_iterator(iterator))
        while True:
            data = []
            for i, batch_size in enumerate(batch_sizes):
                generator = index_generators[i]
                for _ in range(batch_size):
                    idx = next(generator)
                    data.append(dataset[idx])
            yield dataset._collate_fn(data)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfSegDataset(ClfDataset):
    def __getitem__(self, item):
        path = INFO.loc[self.index[item], 'path']
        # print(path)
        with np.load(path) as npz:
            if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['voxel'], npz['seg'])
            elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['raw'], npz['nodule_mask'])
        label = self.label[item]
        return voxel, (self.define_label(label), seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


class ClfSigSegDataset(ClfDataset):
    def __init__(self, crop_size=32, move=3, subset=[0, 1, 2, 3], maglinant_subset=[0, 1, 2, 3], shuffle=False,
                 mixup=False, define_label=lambda l: l):
        index = []
        m_index = []
        for sset in subset:
            index += list(INFO[INFO[invasion] == sset].index)
        self.index = tuple(sorted(index))  # the index in the info

        for sset in maglinant_subset:
            m_index += list(INFO[INFO[maglinant] == sset].index)
        self.m_index = tuple(sorted(m_index))  # the index in the info

        self.label = np.array([[label == s for label in LABEL] for s in INFO.loc[self.index, LABEL_NAME]])
        self.transform = Transform(crop_size, move)
        self.define_label = define_label
        self.shuffle = shuffle
        self.mixup = mixup

    # def __getitem__(self, item):
    #     path = INFO.loc[self.index[item], 'path']
    #     # malignant_label = 0
    #     malignant_label = INFO.loc[self.index[item], 'maglinant']
    #     # print(self.index[item], path)
    #     with np.load(path) as npz:
    #         if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
    #             voxel, seg = self.transform(npz['voxel'], npz['seg'])
    #         elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
    #             voxel, seg = self.transform(npz['raw'], npz['nodule_mask'])
    #     label = self.label[item]
    #     return voxel, (self.define_label(label), seg, malignant_label)
    #
    # def get_maglinant_item(self, item):
    #     path = INFO.loc[item, 'path']
    #     malignant_label = int(INFO.loc[item, 'maglinant'])
    #     # print(item, path)
    #     with np.load(path) as npz:
    #         if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
    #             voxel, seg = self.transform(npz['voxel'], npz['seg'])
    #         elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
    #             voxel, seg = self.transform(npz['raw'], npz['nodule_mask'])
    #
    #     return voxel, ([False] * len(LABEL), seg, malignant_label)

    # only pixel segmentation used
    def __getitem__(self, item):
        path = INFO.loc[self.index[item], 'path']
        malignant_label = INFO.loc[self.index[item], 'maglinant']
        # print(self.index[item], path)
        with np.load(path) as npz:
            if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['voxel'], npz['seg'])
            elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
                voxel = self.transform(npz['raw'])
                seg = np.zeros(voxel.shape)
        label = self.label[item]
        return voxel, (self.define_label(label), seg, malignant_label)

    # only pixel segmentation used
    def get_maglinant_item(self, item):
        path = INFO.loc[item, 'path']
        malignant_label = int(INFO.loc[item, 'maglinant'])
        # print(item, path)
        with np.load(path) as npz:
            if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['voxel'], npz['seg'])
            elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
                voxel = self.transform(npz['raw'])
                seg = np.zeros(voxel.shape)

        return voxel, ([False] * len(LABEL), seg, malignant_label)


    @classmethod
    def get_balanced_loader(cls, batch_sizes, m_batch_sizes, *args, **kwargs):
        assert len(batch_sizes) == len(LABEL)
        dataset = cls(*args, **kwargs)
        invasion_size = len(dataset)
        # print('Size', invasion_size)
        index_generators = []
        m_index_generators = []
        for l_idx in range(len(batch_sizes)):
            # this must be list, or `l_idx` will not be eval
            iterator = [i for i in range(invasion_size) if dataset.label[i, l_idx]]
            index_generators.append(shuffle_iterator(iterator))
        for m_inx in range(len(m_batch_sizes)):
            iterator = [i for i in dataset.m_index if int(INFO.loc[i, 'maglinant']) == m_inx]
            m_index_generators.append(shuffle_iterator(iterator))
        while True:
            data = []
            for i, batch_size in enumerate(batch_sizes):
                generator = index_generators[i]
                for _ in range(batch_size):
                    idx = next(generator)
                    data.append(dataset[idx])
            for i, batch_size in enumerate(m_batch_sizes):
                generator = m_index_generators[i]
                for _ in range(batch_size):
                    idx = next(generator)
                    data.append(dataset.get_maglinant_item(idx))
            yield dataset._collate_fn(data, dataset.mixup, dataset.shuffle)

    @staticmethod
    def _collate_fn(data, mixup=False, shuffle=False, beta=0.2):
        xs = []
        ys = []
        segs = []
        sigs = []
        for index, (x, y) in enumerate(data):
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
            sigs.append(y[2])
        input_arr = np.array(xs)
        clf_arr = np.array(ys)
        seg_arr = np.array(segs)
        sig_arr = np.expand_dims(sigs, axis=-1)

        if mixup:  # mixup must be first
            invasion_label_index = np.random.permutation(np.where(clf_arr.sum(axis=-1))[0])
            maglinant_label_index = np.random.permutation(np.where(clf_arr.sum(axis=-1) == 0)[0])
            random_index = invasion_label_index.tolist() + maglinant_label_index.tolist()
            _lambda = np.random.beta(beta, beta)
            input_arr = _lambda*input_arr + (1-_lambda)*input_arr[random_index]
            clf_arr = _lambda*clf_arr + (1-_lambda)*clf_arr[random_index]
            seg_arr = _lambda*seg_arr + (1-_lambda)*seg_arr[random_index]
            sig_arr = _lambda*sig_arr + (1-_lambda)*sig_arr[random_index]

        if shuffle:
            random_index = np.random.permutation(len(data))
            input_arr = input_arr[random_index]
            clf_arr = clf_arr[random_index]
            seg_arr = seg_arr[random_index]
            sig_arr = sig_arr[random_index]
        return input_arr, {"clf": clf_arr, "seg": seg_arr, "sig": sig_arr}


class BinaryDataset(ClfDataset):
    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        with np.load(PATH.get_nodule(name)) as npz:
            voxel = self.transform(npz['voxel'])
        label = self.label[item]
        return voxel, (label[2] + label[3])


class BinaryNBGDataset(ClfDataset):
    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        with np.load(PATH.get_nodule(name)) as npz:
            voxel = self.transform(npz['voxel'] * npz['seg'])
        label = self.label[item]
        return voxel, (label[2] + label[3])


class BinarySegDataset(ClfDataset):
    def __getitem__(self, item):
        path = INFO.loc[self.index[item], 'path']
        # print(path)
        with np.load(path) as npz:
            if not len(list(set(['voxel', 'seg']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['voxel'], npz['seg'])
            elif not len(list(set(['raw', 'nodule_mask']) ^ set(npz.files))):
                voxel, seg = self.transform(npz['raw'], npz['nodule_mask'])
        label = self.label[item]
        return voxel, ((label[2] + label[3]), seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


class Transform:
    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)


if __name__ == "__main__":
    SIZE = [40, 40, 40]
    train_loader = ClfSigSegDataset.get_balanced_loader(batch_sizes=[2, 2, 2, 2, 2], m_batch_sizes=[4, 4],
                                                        crop_size=SIZE, subset=[0, 1, 2, 3],
                                                        maglinant_subset=[0, 1, 2, 3], move=None,
                                                        mixup=True, shuffle=False)
    val_loader = ClfSigSegDataset.get_balanced_loader(batch_sizes=[2, 2, 2, 2, 2], m_batch_sizes=[4, 4],
                                                      crop_size=SIZE, subset=[4], maglinant_subset=[4], move=None)
    v, a = next(train_loader)

