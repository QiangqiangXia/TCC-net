import os
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PARAMS = {'cifar10':{'epochs':120, 'batch_size':128, 'num_classes':10},
         'cifar100':{'epochs':120, 'batch_size':128, 'num_classes':100},
         'animal10N':{'epochs':100, 'batch_size':128, 'num_classes':10},
         'clothing1M':{'epochs':15, 'batch_size':128, 'num_classes':14}}

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_lr_scheduler(dataset):
    if dataset == 'cifar10':
        def lr_scheduler_cifar10(epoch):
            if epoch < 40:
                return 0.01 
            elif 40 <= epoch < 80:
                return 0.01
            else:
                return 0.001
        return lr_scheduler_cifar10

    elif dataset == 'cifar100':
        def lr_scheduler_cifar100(epoch):
            if epoch < 40:
                return 0.01 
            elif 40 <= epoch < 80:
                return 0.01
            else:
                return 0.001
        return lr_scheduler_cifar100
    
    elif dataset == 'animal10N':
        def lr_scheduler_animal10N(epoch):
            if epoch < 40:
                return 1e-2
            elif 40 <= epoch < 60:
                return 1e-1
            elif 60 <= epoch < 80:
                return 1e-2
            else:
                return 1e-3
        return lr_scheduler_animal10N
    
    else:
        def lr_scheduler_clothing1M(epoch):
            if epoch < 5:
                return 1e-3
            elif 5 <= epoch < 10:
                return 1e-3
            else:
                return 1e-4
        return lr_scheduler_clothing1M

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.percentage = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        self.percentage = self.avg*100