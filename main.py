import os
import time
import numpy as np
from collections import OrderedDict
import shutil

from utils import *
from meta import *
import data_processing.cifar10 as dataset
#import data_processing.cifar100 as dataset
#import data_processing.animal10N as dataset
#import data_processing.clothing1M as dataset
import model_getter as model

# pytorch imports
import torch
import torch.nn as nn
from torchvision import transforms,datasets
import torch.utils.data as data
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

PARAMS_META = {'cifar10':{'stage1':40,'stage2':120}}
#PARAMS_META = {'cifar100':{'stage1':40,'stage2':120}}
#PARAMS_META = {'animal10N':{'stage1':40,'stage2':100}}
#PARAMS_META = {'clothing1M':{'stage1':5,'stage2':15}}

def metapencil(stage1, stage2):

    def meta_training(vnet1, vnet2):

        meta_model1 = model.CNN().to(device)
        meta_model1.load_state_dict(net1.state_dict())
        meta_model1.train()
        output1 = meta_model1(images_meta22)
        lc1 = F.cross_entropy(output1, labels_meta22, reduction='none')
        lc_v1 = torch.reshape(lc1, (len(lc1), 1))
        vnet_v1 = vnet1(lc_v1.data)
        l_f1 = torch.mean(lc_v1 * vnet_v1)
        del output1

        grads1 = torch.autograd.grad(l_f1, meta_model1.parameters(), create_graph=True)
        meta_model1_optimizer = MetaSGD(meta_model1, meta_model1.parameters(), lr=lr)
        meta_model1_optimizer.load_state_dict(optimizer1.state_dict())
        meta_model1_optimizer.meta_step(grads1)
        del grads1
        
        fast_out1 = meta_model1.forward(images_meta1)
        lc1_meta = criterion_cce(fast_out1, labels_meta1)

        optimizer_vnet1.zero_grad()
        lc1_meta.backward()
        optimizer_vnet1.step()

        output1 = net1(images_meta22)
        cost1 = F.cross_entropy(output1, labels_meta22, reduction='none')
        cost_v1 = torch.reshape(cost1, (len(cost1), 1))
        with torch.no_grad():
            vnet_w1 = vnet1(cost_v1)
        loss1 = torch.mean(cost_v1 * vnet_w1)
        del output1

        meta_model2 = model.CNN().to(device)
        meta_model2.load_state_dict(net2.state_dict())
        meta_model2.train()
        output2 = meta_model2(images_meta11)
        lc2 = F.cross_entropy(output2, labels_meta11, reduction='none')
        lc_v2 = torch.reshape(lc2, (len(lc2), 1))
        vnet_v2 = vnet2(lc_v2.data)
        l_f2 = torch.mean(lc_v2 * vnet_v2)
        del output2

        grads2 = torch.autograd.grad(l_f2, meta_model2.parameters(), create_graph=True)
        meta_model2_optimizer = MetaSGD(meta_model2, meta_model2.parameters(), lr=lr)
        meta_model2_optimizer.load_state_dict(optimizer2.state_dict())
        meta_model2_optimizer.meta_step(grads2)
        del grads2
        
        fast_out2 = meta_model2.forward(images_meta2)
        lc2_meta = criterion_cce(fast_out2, labels_meta2)

        optimizer_vnet2.zero_grad()
        lc2_meta.backward()
        optimizer_vnet2.step()

        output2 = net2(images_meta11)
        cost2 = F.cross_entropy(output2, labels_meta11, reduction='none')
        cost_v2 = torch.reshape(cost2, (len(cost2), 1))
        with torch.no_grad():
            vnet_w2 = vnet2(cost_v2)
        loss2 = torch.mean(cost_v2 * vnet_w2)
        del output2

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        return loss1, loss2

    criterion_cce = nn.CrossEntropyLoss()   
    
    start_epoch = 0
    
    best_acc1 = 0
    test_acc_best1 = 0
    epoch_best1 = 0
    
    best_acc2 = 0
    test_acc_best2 = 0
    epoch_best2 = 0
          
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        net1.load_state_dict(checkpoint['state_dict1'])
        net2.load_state_dict(checkpoint['state_dict2'])
        best_acc1 = checkpoint['best_acc1']
        best_acc2 = checkpoint['best_acc2']
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])

    for epoch in range(start_epoch, stage2):
      
        start_epoch = time.time()
      
        train_loss1 = AverageMeter()
        train_loss2 = AverageMeter()
        
        label_similarity = AverageMeter()
        label_similarity1 = AverageMeter()
        label_similarity11 = AverageMeter()
        label_similarity2 = AverageMeter()
        label_similarity22 = AverageMeter()
        
        lr = lr_scheduler(epoch)
        set_learningrate(optimizer1, lr)
        set_learningrate(optimizer2, lr)
        
        net1.train()
        net2.train()
        vnet1.train()
        vnet2.train()
    
        if epoch < stage1:

            for batch_idx, (images, labels, index, true_labels) in enumerate(train_dataloader):

                # training images and labels
                images, labels, true_labels = images.to(device), labels.to(device), true_labels.to(device)
                
                # compute output           
                output1 = net1(images)
                loss1 = CLoss(a=(1-noise_ratio),b=noise_ratio,c=0.1)(output1, labels)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()

                output2 = net2(images)
                loss2 = CLoss(a=(1-noise_ratio),b=noise_ratio,c=0.1)(output2, labels)
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()

                train_loss1.update(loss1.item())
                train_loss2.update(loss2.item())
                
                label_similarity.update(true_labels.eq(labels).cpu().sum().item(), labels.size(0))
                
            test_accuracy1, test_loss1 = evaluate(net1, test_dataloader, criterion_cce)
            test_accuracy2, test_loss2 = evaluate(net2, test_dataloader, criterion_cce)
        
            if test_accuracy1 > test_acc_best1:
                test_acc_best1 = test_accuracy1
                epoch_best1 = epoch + 1
            
            if test_accuracy2 > test_acc_best2:
                test_acc_best2 = test_accuracy2
                epoch_best2 = epoch + 1
            
            is_best1 = test_accuracy1 > best_acc1
            best_acc1 = max(test_accuracy1, best_acc1)
            is_best2 = test_accuracy2 > best_acc2
            best_acc2 = max(test_accuracy2, best_acc2)
            save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict1': net1.state_dict(),
                  'state_dict2': net2.state_dict(),
                  'acc1': test_accuracy1,
                  'acc2': test_accuracy2,
                  'best_acc1': best_acc1,
                  'best_acc2': best_acc2,
                  'optimizer1' : optimizer1.state_dict(),
                  'optimizer2' : optimizer2.state_dict(),
                }, is_best1, is_best2)

            summary_writer.add_scalar('train_loss1', train_loss1.avg, epoch + 1)
            summary_writer.add_scalar('test_loss1', test_loss1, epoch + 1)
            summary_writer.add_scalar('test_accuracy1', test_accuracy1, epoch + 1)
            summary_writer.add_scalar('test_accuracy_best1', test_acc_best1, epoch + 1)                      
            summary_writer.add_scalar('train_loss2', train_loss2.avg, epoch + 1)
            summary_writer.add_scalar('test_loss2', test_loss2, epoch + 1)
            summary_writer.add_scalar('test_accuracy2', test_accuracy2, epoch + 1)
            summary_writer.add_scalar('test_accuracy_best2', test_acc_best2, epoch + 1)

            template = 'Epoch {}, Accuracy1(test):{:3.1f}, Accuracy2(test):{:3.1f}, Loss1(train,test):{:4.3f}/{:4.3f}, Loss2(train,test):{:4.3f}/{:4.3f}, Label similarity:{:6.3f}, Learning rate(lr):{}, Time:{:3.1f}'
            print(template.format(epoch + 1, 
                            test_accuracy1,test_accuracy2,   
                            train_loss1.avg, test_loss1,
                            train_loss2.avg, test_loss2,
                            label_similarity.percentage,
                            lr, time.time()-start_epoch))

        elif stage1 <= epoch < stage2:

            #forget_rates = [0, 0.08, 0.16, 0.24, 0.32, 0.4, 0.4, 0.4, 0.4, 0.4] for Clothing1M
            for batch_idx, (images, labels, index, true_labels) in enumerate(train_dataloader):
                
                loss_values1 = np.zeros(len(labels))
                loss_values2 = np.zeros(len(labels)) 
                
                num_meta1 = int(0.5*(1-noise_ratio)*len(labels))
                num_meta2 = int((1-noise_ratio)*len(labels))
                #num_meta1 = int(0.5*(1-forget_rates[epoch-stage1])*len(labels)) for Clothing1M
                #num_meta2 = int((1-forget_rates[epoch-stage1])*len(labels)) for Clothing1M

                images, labels = images.to(device), labels.to(device)

                output1 = net1(images)
                loss1 = F.cross_entropy(output1, labels, reduction='none')
                loss_values1 = loss1.detach().cpu().numpy()
                idx_meta1 = np.argsort(loss_values1)[:num_meta1]
                idx_meta11 = np.argsort(loss_values1)[:num_meta2]
                
                output2 = net2(images)
                loss2 = F.cross_entropy(output2, labels, reduction='none')  
                loss_values2 = loss2.detach().cpu().numpy()
                idx_meta2 = np.argsort(loss_values2)[:num_meta1]
                idx_meta22 = np.argsort(loss_values2)[:num_meta2]

                images_meta1, labels_meta1, true_labels_meta1 = images[idx_meta1, :].to(device), labels[idx_meta1].to(device), true_labels[idx_meta1].to(device)
                images_meta11, labels_meta11, true_labels_meta11 = images[idx_meta11, :].to(device), labels[idx_meta11].to(device), true_labels[idx_meta11].to(device)
                
                images_meta2, labels_meta2, true_labels_meta2 = images[idx_meta2, :].to(device), labels[idx_meta2].to(device), true_labels[idx_meta2].to(device)
                images_meta22, labels_meta22, true_labels_meta22 = images[idx_meta22, :].to(device), labels[idx_meta22].to(device), true_labels[idx_meta22].to(device)
     
                loss1, loss2 = meta_training(vnet1, vnet2)
                
                train_loss1.update(loss1.item())
                train_loss2.update(loss2.item())
                
                label_similarity1.update(true_labels_meta1.eq(labels_meta1).cpu().sum().item(), labels_meta1.size(0))
                label_similarity11.update(true_labels_meta11.eq(labels_meta11).cpu().sum().item(), labels_meta11.size(0))
                label_similarity2.update(true_labels_meta2.eq(labels_meta2).cpu().sum().item(), labels_meta2.size(0))
                label_similarity22.update(true_labels_meta22.eq(labels_meta22).cpu().sum().item(), labels_meta22.size(0))
                
            test_accuracy1, test_loss1 = evaluate(net1, test_dataloader, criterion_cce)
            test_accuracy2, test_loss2 = evaluate(net2, test_dataloader, criterion_cce)

            if test_accuracy1 > test_acc_best1:
                test_acc_best1 = test_accuracy1
                epoch_best1 = epoch + 1
                
            if test_accuracy2 > test_acc_best2:
                test_acc_best2 = test_accuracy2
                epoch_best2 = epoch + 1

            is_best1 = test_accuracy1 > best_acc1
            best_acc1 = max(test_accuracy1, best_acc1)
            is_best2 = test_accuracy2 > best_acc2
            best_acc2 = max(test_accuracy2, best_acc2)
            save_checkpoint({
                  'epoch': epoch + 1,
                  'state_dict1': net1.state_dict(),
                  'state_dict2': net2.state_dict(),
                  'acc1': test_accuracy1,
                  'acc2': test_accuracy2,
                  'best_acc1': best_acc1,
                  'best_acc2': best_acc2,
                  'optimizer1' : optimizer1.state_dict(),
                  'optimizer2' : optimizer2.state_dict(),
                }, is_best1, is_best2)

            summary_writer.add_scalar('train_loss1', train_loss1.avg, epoch + 1)
            summary_writer.add_scalar('test_loss1', test_loss1, epoch + 1)
            summary_writer.add_scalar('test_accuracy1', test_accuracy1, epoch + 1)
            summary_writer.add_scalar('test_accuracy_best1', test_acc_best1, epoch + 1)
            summary_writer.add_scalar('train_loss2', train_loss2.avg, epoch + 1)
            summary_writer.add_scalar('test_loss2', test_loss2, epoch + 1)
            summary_writer.add_scalar('test_accuracy2', test_accuracy2, epoch + 1)
            summary_writer.add_scalar('test_accuracy_best2', test_acc_best2, epoch + 1)
         
            template = 'Epoch {}, Accuracy1(test):{:3.1f}, Accuracy2(test):{:3.1f}, Loss1(train,test):{:4.3f}/{:4.3f}, Loss2(train,test):{:4.3f}/{:4.3f}, Label similarity1:{:6.3f}, Label similarity11:{:6.3f}, Label similarity2:{:6.3f}, Label similarity22:{:6.3f}, Learning rate(lr):{}, Time:{:3.1f}'
            print(template.format(epoch + 1, 
                            test_accuracy1,test_accuracy2,   
                            train_loss1.avg, test_loss1,
                            train_loss2.avg, test_loss2,
                            label_similarity1.percentage, label_similarity11.percentage, label_similarity2.percentage, label_similarity22.percentage,  
                            lr, time.time()-start_epoch))

    print('{}({}): Test acc1: {:3.1f}, Test acc2: {:3.1f}, Best epoch1: {}, Best acc1: {}, Best epoch2: {}, Best acc2: {}'.format(
        noise_type,noise_ratio,test_acc_best1,test_acc_best2,epoch_best1,best_acc1,epoch_best2,best_acc2))
    
    summary_writer.close()
    hp_writer.add_hparams({'stage1':stage1, 'stage2':stage2}, 
            {'test_accuracy_best1': test_acc_best1, 'test_accuracy_best2': test_acc_best2, 'epoch_best1': epoch_best1, 'epoch_best2': epoch_best2})
    hp_writer.close()
    torch.save(net1.state_dict(), os.path.join(log_dir, 'saved_model1.pt'))
    torch.save(net2.state_dict(), os.path.join(log_dir, 'saved_model2.pt'))

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()

    net.eval()
    with torch.no_grad():
        for (inputs, targets) in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            loss = criterion(outputs, targets) 
            _, predicted = torch.max(outputs.data, 1) 
            eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            eval_loss.update(loss.item())
    return eval_accuracy.percentage, eval_loss.avg

class CLoss(torch.nn.Module):
    def __init__(self, a, b, c):
        super(CLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        
        # l1
        l1 = self.cross_entropy(outputs, labels)
       
        # l2
        pred = F.softmax(outputs, dim=1)
        pred = torch.log(1-pred)      
        l2 = F.nll_loss(pred,labels)

        #l3
        pred = F.softmax(outputs, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, 10).float().to(device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-34, max=1.0) #min=1e-10 for Cifar100; min=1e-1 for animal10N; min=1e-45 for Clothing1M
        l3 = -torch.mean(torch.sum(pred * torch.log(label_one_hot), dim=1))
        
        # Loss
        loss = self.a * l1 + self.b * l2 + self.c * l3
        return loss
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', required=False, type=str, default='cifar10',
        help="Dataset to use; either 'cifar10', 'cifar100', 'animal10n', 'clothing1M'")
    parser.add_argument('-n', '--noise_type', required=False, type=str, default='symmetric',
        help="Noise type for cifar10: 'asymmetric', 'symmetric'")
    parser.add_argument('-s', '--batch_size', required=False, type=int,
        help="Number of gpus to be used")
    parser.add_argument('-i', '--gpu_ids', required=False, type=int, nargs='+', action='append',
        help="GPU ids to be used")
    parser.add_argument('-f', '--folder_log', required=False, type=str,
        help="Folder name for logs")
    parser.add_argument('-w', '--num_workers', required=False, type=int, default=4,
        help="Number of parallel workers to parse dataset")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")

    parser.add_argument('--momentum', required=False, type=float, default=0.9)
    parser.add_argument('--dampening', required=False, type=float, default=0)
    parser.add_argument('--nesterov', required=False, type=bool, default=False)
    parser.add_argument('--weight_decay', required=False, type=float, default=1e-3)
    parser.add_argument('--meta_lr', required=False, type=float, default=1e-3)
    parser.add_argument('--meta_weight_decay', required=False, type=float, default=1e-4)
    
    parser.add_argument('--percent', type=float, default=0.2,
        help='Percentage of noise')
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('--train_ratio', type=float, default=1.0,
        help='Percentage of train')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='./TCC-net/result',
        help='Directory to output the result')

    args = parser.parse_args()

    #set default variables if they are not given from the command line
    if args.stage1 == None: args.stage1 = PARAMS_META[args.dataset_name]['stage1']
    if args.stage2 == None: args.stage2 = PARAMS_META[args.dataset_name]['stage2']
        
    # configuration variables
    dataset_name = args.dataset_name
    model_name = 'TCC-net'
    noise_type = args.noise_type
    noise_ratio = args.percent
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[dataset_name]['batch_size']
    NUM_CLASSES = PARAMS[dataset_name]['num_classes']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    create_folder(args.out)
    
    if args.gpu_ids is None:
        ngpu = torch.cuda.device_count() if device.type == 'cuda' else 0  
        gpu_ids = list(range(ngpu)) 
    else:
        gpu_ids = args.gpu_ids[0]
        ngpu = len(gpu_ids)
        if ngpu == 1: 
            device = torch.device("cuda:{}".format(gpu_ids[0]))
        
    if args.num_workers is None:
        num_workers = 2 if ngpu < 2 else ngpu*2
    else:
        num_workers = args.num_workers
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    train_dataset, val_dataset = dataset.get_cifar10('./TCC-net/dataset', args, train=True, download=False, transform_train=transform_train, transform_val=transform_val)
    test_dataset = datasets.CIFAR10('./TCC-net/dataset', train=False, transform=transform_val)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.492, 0.482, 0.446), (0.247, 0.244, 0.262)),
    ])
    
    '''for Animal10N
    train_dataset = Animal10(split='train', transform=transform_train)
    test_dataset = Animal10(split='test', transform=transform_test)
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])'''
    
    ''' for Clothing1M
    loader = dataset.clothing_dataloader(batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)
    train_dataloader, test_dataloader = loader.run()'''

    net1 = model.CNN().to(device)
    net2 = model.CNN().to(device)
    '''for Clothing1M
    net1 = model.resnet18().to(device) 
    net2 = model.resnet18().to(device)'''
    vnet1 = model.VNet().to(device)
    vnet2 = model.VNet().to(device)
    
    if (device.type == 'cuda') and (ngpu > 1):
        net1 = nn.DataParallel(net1, list(range(ngpu)))
        net2 = nn.DataParallel(net2, list(range(ngpu)))
        vnet1 = nn.DataParallel(vnet1, list(range(ngpu)))
        vnet2 = nn.DataParallel(vnet2, list(range(ngpu)))

    lr_scheduler = get_lr_scheduler(dataset_name)

    optimizer1 = optim.SGD(net1.parameters(), lr=lr_scheduler(0), momentum=args.momentum, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer2 = optim.SGD(net2.parameters(), lr=lr_scheduler(0), momentum=args.momentum, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer_vnet1 = optim.Adam(vnet1.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)
    optimizer_vnet2 = optim.Adam(vnet2.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay)

    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    softmax = nn.Softmax(dim=1).to(device)
    
    print("Noise type: {}, Noise ratio: {}".format(noise_type, noise_ratio))

    # if logging
    base_folder = noise_type + '/' + str(noise_ratio) + '/' + model_name
    log_folder = args.folder_log if args.folder_log else 's{}_{}'.format(args.stage1,current_time)
    log_base = './TCC-net/{}/logs/{}/'.format(dataset_name, base_folder)
    log_dir = log_base + log_folder + '/'
    log_dir_hp = './TCC-net/{}/logs_hp/{}/'.format(dataset_name, base_folder)
    create_folder(log_dir)
    summary_writer = SummaryWriter(log_dir)
    create_folder(log_dir_hp)
    hp_writer = SummaryWriter(log_dir_hp)
    
    def save_checkpoint(state, is_best1, is_best2, checkpoint=args.out, filename='checkpoint.pth.tar'):
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        if is_best1:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best1.pth.tar'))
        if is_best2:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best2.pth.tar'))
    
    start_train = time.time()
    metapencil(args.stage1, args.stage2)
    print('Total training duration: {:3.2f}h'.format((time.time()-start_train)/3600))