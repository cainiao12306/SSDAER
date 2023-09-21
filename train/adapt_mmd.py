"""Adversarial adaptation to train target encoder."""
import sys
sys.path.append('../')
import torch
from utils import make_cuda
import torch.nn.functional as F
import torch.nn as nn
import param
import torch.optim as optim
from utils import save_model,enable_dropout
import csv
import os
from metrics import mmd
import numpy as np
from torch.utils.data import DataLoader,ConcatDataset,TensorDataset,RandomSampler
from sklearn.cluster import KMeans
import itertools
import matplotlib.pyplot as plt

def warmtrain(args, encoder, classifier, src_data_loader, tgt_data_train_loader, tgt_data_valid_loader):
    """Train encoder for target domain."""

    # set train state for Dropout and BN layers
    #如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。
    #对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    encoder.train()
    classifier.train()
    # setup criterion and optimizer
    BCELoss = nn.BCELoss()#二分类交叉熵损失
    CELoss = nn.CrossEntropyLoss()
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')#KL散度
    optimizer = optim.Adam(list(encoder.parameters())+list(classifier.parameters()), lr=param.d_learning_rate)
    #len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))
    bestf1 = 0.0
    besttrainf1 = 0.0
    for epoch in range(args.num_epochs):
        if len(src_data_loader)>len(tgt_data_train_loader):
            data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        else:
            data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
        mmd_sum = 0
        jssum = 0
        len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))
        for step, (src, tgt) in data_zip:
            if tgt:
                reviews_src, src_mask,src_segment, labels,_ = src
                reviews_tgt, tgt_mask,tgt_segment, _,_ = tgt
                reviews_src = make_cuda(reviews_src)
                src_mask = make_cuda(src_mask)
                src_segment = make_cuda(src_segment)
                labels = make_cuda(labels)
                reviews_tgt = make_cuda(reviews_tgt)
                tgt_mask = make_cuda(tgt_mask)
                tgt_segment = make_cuda(tgt_segment)
    
                # zero gradients for optimizer
                optimizer.zero_grad()
    
                # extract and concat features
                feat_src = encoder(reviews_src, src_mask, src_segment)
                feat_tgt = encoder(reviews_tgt, tgt_mask, tgt_segment)
                preds = classifier(feat_src)
                cls_loss = CELoss(preds, labels)
                loss_mmd = mmd.mmd_rbf_noaccelerate(feat_src, feat_tgt)
                p = float(step + epoch * len_data_loader) / args.num_epochs / len_data_loader
                lamda = 2. / (1. + np.exp(-10 * p)) - 1
                if args.source_only:
                    loss = cls_loss
                else:
                    loss = cls_loss + args.beta * loss_mmd
            else:
                reviews_src, src_mask,src_segment, labels = src
                reviews_src = make_cuda(reviews_src)
                src_mask = make_cuda(src_mask)
                src_segment = make_cuda(src_segment)
                labels = make_cuda(labels)
                optimizer.zero_grad()
    
                # extract and concat features
                feat_src = encoder(reviews_src, src_mask, src_segment)
                preds = classifier(feat_src)
                cls_loss = CELoss(preds, labels)
                p = float(step + epoch * len_data_loader) / args.num_epochs / len_data_loader
                lamda = 2. / (1. + np.exp(-10 * p)) - 1
                loss = cls_loss
            loss.backward()
            optimizer.step()
            mmd_sum += loss_mmd.item()
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "mmd_loss=%.4f cls_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         loss_mmd.item(),
                         cls_loss.item()))
        f1_train,eloss_train = evaluate(args, encoder, classifier, tgt_data_train_loader, src_data_loader,epoch=epoch, pattern=1000)
        f1_valid,eloss_valid = evaluate(args, encoder, classifier, tgt_data_valid_loader, src_data_loader,epoch=epoch, pattern=1000)
        if f1_valid>bestf1:
            save_model(args, encoder, param.src_encoder_path + 'mmdbestmodel')
            save_model(args, classifier, param.src_classifier_path + 'mmdbestmodel')
            bestf1 = f1_valid
            besttrainf1 = f1_train

    return encoder,classifier,besttrainf1


def train(args,encoder, classifier, src_data_loader, tgt_data_train_loader, tgt_data_valid_loader,pseudo_data_loader):
    """Train encoder for target domain."""
    encoder.train()
    classifier.train()
    # setup criterion and optimizer
    BCELoss = nn.BCELoss()  # 二分类交叉熵损失
    CELoss = nn.CrossEntropyLoss()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # KL散度
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=param.d_learning_rate)
    # len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))
    bestf1 = 0.0
    besttrainf1 = 0.0
    for epoch in range(args.num_epochs):

        data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader, pseudo_data_loader))
        mmd_sum = 0
        jssum = 0
        len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader), len(pseudo_data_loader))
        for step, (src, tgt, pse) in data_zip:
            reviews_src, src_mask, src_segment, labels, _ = src
            reviews_tgt, tgt_mask, tgt_segment, _, _ = tgt
            reviews_pse, pse_mask, pse_segment, pse_labels,_ = pse
            reviews_src = make_cuda(reviews_src)
            src_mask = make_cuda(src_mask)
            src_segment = make_cuda(src_segment)
            labels = make_cuda(labels)
            reviews_tgt = make_cuda(reviews_tgt)
            tgt_mask = make_cuda(tgt_mask)
            tgt_segment = make_cuda(tgt_segment)
            reviews_pse = make_cuda(reviews_pse)
            pse_mask = make_cuda(pse_mask)
            pse_segment = make_cuda(pse_segment)
            pse_labels = make_cuda(pse_labels)


            # zero gradients for optimizer
            optimizer.zero_grad()

            # extract and concat features
            feat_src = encoder(reviews_src, src_mask, src_segment)
            feat_tgt = encoder(reviews_tgt, tgt_mask, tgt_segment)
            feat_pse = encoder(reviews_pse, pse_mask, pse_segment)
            feat_pse, pse_targets_a, pse_targets_b, lam = mixup_data(feat_pse,pse_labels, alpha=1.0,device='cuda')

            src_preds = classifier(feat_src)
            pse_preds = classifier(feat_pse)
            _, pse_loss = loss_mixup_reg_ep(args, pse_preds, pse_targets_a, pse_targets_b, 'cuda', lam)
            src_loss = CELoss(src_preds, labels)
            cls_loss = src_loss + 0.001 * pse_loss
            #cls_loss = src_loss
            loss_mmd = mmd.mmd_rbf_noaccelerate(feat_src, feat_tgt)
            #p = float(step + epoch * len_data_loader) / args.num_epochs / len_data_loader
            #lamda = 2. / (1. + np.exp(-10 * p)) - 1
            if args.source_only:
                loss = cls_loss
            else:
                loss = cls_loss + args.beta * loss_mmd

            loss.backward()
            optimizer.step()
            mmd_sum += loss_mmd.item()
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: "
                      "mmd_loss=%.4f cls_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len_data_loader,
                         loss_mmd.item(),
                         cls_loss.item()))
        f1_train, eloss_train = evaluate(args, encoder, classifier, tgt_data_train_loader, src_data_loader, epoch=epoch,
                                         pattern=1000)
        f1_valid, eloss_valid = evaluate(args, encoder, classifier, tgt_data_valid_loader, src_data_loader, epoch=epoch,
                                         pattern=1000)
        if f1_valid > bestf1:
            #save_model(args, encoder, param.src_encoder_path + 'mmdbestmodel')
            #save_model(args, classifier, param.src_classifier_path + 'mmdbestmodel')
            bestf1 = f1_valid
            besttrainf1 = f1_train

    return encoder, classifier, besttrainf1

def loss_mixup_reg_ep(args, preds, targets_a, targets_b, device, lam):
    num_classes = 2
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
    prob = F.softmax(preds, dim=1)  # 计算预测结果的softmax概率
    prob_avg = torch.mean(prob, dim=0)  # 计算softmax概率的平均值
    p = torch.ones(num_classes).to(device) / num_classes  # 假设先验概率为均匀分布

    #mixup_loss_a = torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))  # 使用混合目标a计算交叉熵损失
    #mixup_loss_b = torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))  # 使用混合目标b计算交叉熵损失
    mixup_loss_a = BCEWithLogitsLoss(preds,targets_a)
    mixup_loss_b = BCEWithLogitsLoss(preds,targets_b)
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b  # 使用混合系数lam进行加权

    L_p = -torch.sum(torch.log(prob_avg) * p)  # 使用先验概率计算交叉熵损失
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))  # 使用熵计算交叉熵损失

    loss = mixup_loss + args.reg1 * L_p + args.reg2 * L_e  # 组合混合损失和正则化的损失
    #loss = mixup_loss

    return prob, loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def evaluate(args, encoder, classifier, data_loader, src_data_loader, flag=None,epoch=None,pattern=None):
    # set eval state for Dropout and BN layers
    encoder.eval()#model.eval()的作用是不启用 Batch Normalization 和 Dropout
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0
    tp = 0
    fp = 0
    p = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    first = 0
    for (reviews, mask,segment, labels,_) in data_loader:
        
        truelen = torch.sum(mask, dim=1)
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        labels = make_cuda(labels)
        
        with torch.no_grad():
            feat = encoder(reviews, mask,segment)    
            preds = classifier(feat)
        loss += criterion(preds, labels).item()
        #print(preds.data)
        #print(preds.data.max(1)[0])
        pred_cls = preds.data.max(1)[1]#troch.max()[1]只返回最大值的每个索引;torch.max()[0]， 只返回最大值的每个数

        #print(pred_cls)
        
        acc += pred_cls.eq(labels.data).cpu().sum().item()
        for i in range(len(labels)):
            if labels[i] == 1:
                p += 1
                if pred_cls[i] == 1:
                    tp += 1
            else:
                if pred_cls[i] == 1:
                    fp += 1
    div_safe = 0.000001
    print("p",p)
    print("tp",tp)
    print("fp",fp)
    recall = tp/(p+div_safe)
    
    precision = tp/(tp+fp+div_safe)
    f1 = 2*recall*precision/(recall + precision + div_safe)
    print("recall",recall)
    print("precision",precision)
    print("f1",f1)

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = %.4f, Avg Accuracy = %.4f" % (loss, acc))

    if flag:
            f = open('res.csv','a',encoding='utf-8',newline="")
            csv_writer = csv.writer(f)
            row = []
            row.append([flag,p,tp,fp,recall,precision,f1])
            csv_writer.writerows(row)
            f.close()

    return f1,loss