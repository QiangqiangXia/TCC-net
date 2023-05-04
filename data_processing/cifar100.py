import numpy as np
from PIL import Image
import torchvision
from numpy.testing import assert_array_almost_equal


def get_cifar100(root, args, train=True, transform_train=None, transform_val=None, download=False):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=False)
    train_idxs, val_idxs = train_val_split(base_dataset.targets, args.train_ratio)
    train_dataset = CIFAR100_train(root, train_idxs, args, train=train, transform=transform_train)

    if args.noise_type == 'symmetric':
        train_dataset.symmetric_noise()
    else:
        train_dataset.asymmetric_noise()

    val_dataset = CIFAR100_val(root, val_idxs, train=train, transform=transform_val)

    return train_dataset, val_dataset


def train_val_split(train_val, train_ratio):
    train_val = np.array(train_val)
    train_n = int(len(train_val) * train_ratio / 100)
    train_idxs = []
    val_idxs = []

    for i in range(100):
        idxs = np.where(train_val == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])

    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR100_train(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, args=None, train=True, transform=None, target_transform=None, download=False):

        super(CIFAR100_train, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.args = args
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.true_labels = self.targets + 1 - 1
        self.count_img = 0

    def symmetric_noise(self):

        indices = np.random.permutation(len(self.data))

        for i, idx in enumerate(indices):
            if i < self.args.percent * len(self.data):
                self.targets[idx] = np.random.randint(100, dtype=np.int32)

    def asymmetric_noise(self, random_state=0):
        nb_classes = 100
        P = np.eye(nb_classes)
        n = self.args.percent
        if n > 0.0:
            P[0, 0], P[0, 1] = 1. - n, n
            for i in range(1, nb_classes - 1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n
            print(P)
            
            assert P.shape[0] == P.shape[1]         
            assert np.max(self.targets) < P.shape[0]   
            assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))    
            assert (P >= 0.0).all()

            m = self.targets.shape[0]
            flipper = np.random.RandomState(random_state)
            for idx in np.arange(m):
                i = self.targets[idx]
                flipped = flipper.multinomial(1, P[i, :], 1)[0]
                self.targets[idx] = np.where(flipped == 1)[0]
                

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.count_img += 1
        img, target = self.data[index], self.targets[index]
        true_labels = self.true_labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, true_labels


class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100_val, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.data = self.data[indexs]
        self.targets = np.array(self.targets)[indexs]