import csv
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler,ConcatDataset
import param
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,label_id=None,exm_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.exm_id = exm_id
class InputFeaturesED(object):
    """A single set of features of data for ED."""
    def __init__(self, input_ids, attention_mask,label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id

def CSV2Array(path):
    """Read data from csv"""
    data = pd.read_csv(path, encoding='latin')
    pairs, labels = data.pairs.values.tolist(), data.labels.values.tolist()
    return pairs, labels

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(args, net, restore=None):
    """ restore model weights """
    if restore is not None:
        path = os.path.join(param.model_root, args.src, args.model, str(args.train_seed), restore)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Restore model from: {}".format(os.path.abspath(path)))

    """ check if cuda is available """
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net

def save_model(args, net, name):
    """Save trained model."""
    folder = os.path.join(param.model_root, args.src, args.model, str(args.train_seed))
    path = os.path.join(folder, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(net.state_dict(), path)
    print("save pretrained model to: {}".format(path))

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def convert_examples_to_features(pairs, labels, max_seq_length, tokenizer,
                                 cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0,csv_writer=None,exp_idx=-1):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        # add ER situation
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : # remove excessively long string
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("too long!")
                    continue
            tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
        if ex_index == exp_idx:
            """This is for recording attention"""
            with open('tokens.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(tokens)
            with open('token_type_ids.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                writer.writerow(segment_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids = segment_ids,
                          label_id=label,
                          exm_id=ex_index))
        if csv_writer != None:
            """Record training data for semi"""
            csv_writer.writerow([ex_index, pair, label])
    return features

def get_data_loader(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_exm_ids = torch.tensor([f.exm_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_label_ids,all_exm_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        """Read all data"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        """Delet the last incomplete epoch"""
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader

def get_data_loaderED(features, batch_size,flag):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,all_label_ids)
    sampler = RandomSampler(dataset)
    if flag == "dev":
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return dataloader


def bart_convert_examples_to_features(pairs, labels, max_seq_length, tokenizer, pad_token=0, cls_token='<s>',sep_token='</s>'):
    features = []
    for ex_index, (pair, label) in enumerate(zip(pairs, labels)):
        if (ex_index + 1) % 200 == 0:
            print("writing example %d of %d" % (ex_index + 1, len(pairs)))
        if sep_token in pair:
            left = pair.split(sep_token)[0]
            right = pair.split(sep_token)[1]
            ltokens = tokenizer.tokenize(left)
            rtokens = tokenizer.tokenize(right)
            more = len(ltokens) + len(rtokens) - max_seq_length + 3
            if more > 0:
                if more <len(rtokens) : #从rtokens中删除多余的部分
                    rtokens = rtokens[:(len(rtokens) - more)]
                elif more <len(ltokens):
                    ltokens = ltokens[:(len(ltokens) - more)]
                else:
                    print("bad example!")
                    continue
            tokens =  [cls_token] +ltokens + [sep_token] + rtokens + [sep_token]
            segment_ids = [0]*(len(ltokens)+2) + [1]*(len(rtokens)+1)
        else:
            tokens = tokenizer.tokenize(pair)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [cls_token] + tokens + [sep_token]
            segment_ids = [0]*(len(tokens))
            
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        features.append(InputFeaturesED(input_ids=input_ids,
                        attention_mask=input_mask,
                        label_id=label
                        ))
    return features
  


def MMD(source, target):
    """Compute MMD"""
    mmd_loss = torch.exp(-1 / (source.mean(dim=0) - target.mean(dim=0)).norm())
    return mmd_loss

def select_high_confidence_samples(args, encoder, classifier, tgt_data_loader,src_data_loader,epoch=None,threshold = None):
    """Select high confidence samples from target domain."""

    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # setup criterion
    CELoss = nn.CrossEntropyLoss()

    # create empty lists for selected samples
    high_confidence_reviews = []
    high_confidence_mask = []
    high_confidence_segment = []
    high_confidence_labels = []
    high_confidence_indices = []
    # iterate through target data loader to get predicted labels and probabilities
    for (reviews, mask, segment, label, indices) in tgt_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)

        # extract features and predict labels
        with torch.no_grad():
            feat = encoder(reviews, mask, segment)
            preds = classifier(feat)

        # select samples with high confidence
        for i in range(len(preds)):
            prob = torch.softmax(preds[i], dim=-1)
            max_prob, max_label = torch.max(prob, dim=-1)
            if max_prob > threshold:
                high_confidence_reviews.append(reviews[i])
                high_confidence_mask.append(mask[i])
                high_confidence_segment.append(segment[i])
                high_confidence_labels.append(max_label)
                high_confidence_indices.append(indices[i])

    # convert selected samples to tensors
    high_confidence_reviews = torch.stack(high_confidence_reviews).cpu()
    high_confidence_mask = torch.stack(high_confidence_mask).cpu()
    high_confidence_segment = torch.stack(high_confidence_segment).cpu()
    high_confidence_labels = torch.stack(high_confidence_labels).cpu()
    high_confidence_indices = torch.stack(high_confidence_indices).cpu()

    high_confidence_dataset = TensorDataset(high_confidence_reviews, high_confidence_mask,high_confidence_segment, high_confidence_labels,high_confidence_indices)

    new_dataset = ConcatDataset([src_data_loader.dataset, high_confidence_dataset])
    new_src_data_loader = DataLoader(new_dataset, batch_size=args.batch_size, drop_last=True)

    return new_src_data_loader

def select_credible_samples(args, encoder, classifier, tgt_data_loader, epoch=None):

    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    enable_dropout(encoder)

    # setup criterion
    CELoss = nn.CrossEntropyLoss()

    # create empty lists for selected samples
    high_confidence_reviews = []
    high_confidence_mask = []
    high_confidence_segment = []
    high_confidence_labels = []
    high_confidence_indices = []

    # iterate through target data loader to get predicted labels and probabilities
    for (reviews, mask, segment, _, indices) in tgt_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)

        # extract features and predict labels
        with torch.no_grad():
            # Get the predictions and uncertainty using Monte Carlo dropout
            for j in range(args.num_mc_samples):
                feat = encoder(reviews, mask, segment)
                preds = classifier(feat)
                if j == 0:
                    all_preds = preds.unsqueeze(0)
                else:
                    all_preds = torch.cat((all_preds, preds.unsqueeze(0)), dim=0)
            mean_preds = all_preds.mean(dim=0)
            variance_preds = all_preds.var(dim=0)
            uncertainty = variance_preds.mean(dim=-1)
            confidence = mean_preds.softmax(dim=-1).max(dim=-1)[0]

        # select samples with high confidence and low uncertainty
        for i in range(len(mean_preds)):
            if confidence[i] > args.confidence_threshold and uncertainty[i] < args.uncertainty_threshold:
                high_confidence_reviews.append(reviews[i])
                high_confidence_mask.append(mask[i])
                high_confidence_segment.append(segment[i])
                high_confidence_labels.append(mean_preds[i])

        high_confidence_indices.extend(indices)

    # convert selected samples to tensors
    high_confidence_reviews = torch.stack(high_confidence_reviews).cpu()
    high_confidence_mask = torch.stack(high_confidence_mask).cpu()
    high_confidence_segment = torch.stack(high_confidence_segment).cpu()
    high_confidence_labels = torch.stack(high_confidence_labels).cpu()
    high_confidence_indices = torch.stack(high_confidence_indices).cpu()

    high_confidence_dataset = TensorDataset(high_confidence_reviews, high_confidence_mask, high_confidence_segment, high_confidence_labels, high_confidence_indices)

    new_src_data_loader = DataLoader(high_confidence_dataset, batch_size=args.batch_size, drop_last=True)

    return new_src_data_loader


def gen_pseudo_labels_by_fold_unfold(args, encoder, classifier, tgt_data_loader, src_data_loader, epoch=None):

    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # get labeled data embeddings
    labeled_embeddings = []
    labeled_labels = []
    for (reviews, mask, segment, label, indices) in src_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        with torch.no_grad():
            feat = encoder(reviews, mask, segment)
            labeled_embeddings.append(feat)
            labeled_labels.append(label)
    labeled_embeddings = torch.cat(labeled_embeddings, dim=0)
    labeled_labels = torch.cat(labeled_labels, dim=0)

    # get unlabeled data embeddings and generate pseudo labels
    unlabeled_embeddings = []
    for (reviews, mask, segment, label, indices) in tgt_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        with torch.no_grad():
            feat = encoder(reviews, mask, segment)
            embeddings = torch.cat([feat, labeled_embeddings], dim=0)
            sim_matrix = torch.mm(embeddings, embeddings.t())
            sim_matrix = F.normalize(sim_matrix, dim=1)
            sim_matrix = sim_matrix[-len(reviews):, :len(labeled_embeddings)]
            sim_matrix = torch.cat([sim_matrix.max(dim=1)[0].unsqueeze(1), sim_matrix], dim=1)
            pseudo_labels = labeled_labels[sim_matrix.argmax(dim=1)]
        for i in range(len(reviews)):
            if pseudo_labels[i] != -1:
                unlabeled_embeddings.append(feat[i])
    unlabeled_embeddings = torch.stack(unlabeled_embeddings)

    # generate dataset with pseudo labels
    pseudo_labels = []
    for i in range(len(unlabeled_embeddings)):
        sim_matrix = F.cosine_similarity(unlabeled_embeddings[i].unsqueeze(0), labeled_embeddings)
        pseudo_label = labeled_labels[sim_matrix.argmax()]
        pseudo_labels.append(pseudo_label)
    pseudo_labels = torch.stack(pseudo_labels)
    high_confidence_dataset = TensorDataset(tgt_data_loader.dataset.tensors[0],
                                            tgt_data_loader.dataset.tensors[1],
                                            tgt_data_loader.dataset.tensors[2],
                                            pseudo_labels,
                                            tgt_data_loader.dataset.tensors[4])

    # concatenate datasets and create data loader
    new_dataset = ConcatDataset([src_data_loader.dataset, high_confidence_dataset])
    new_src_data_loader = DataLoader(new_dataset, batch_size=args.batch_size, drop_last=True)

    return new_src_data_loader


def gen_pseudo_labels(args, encoder, classifier, tgt_data_loader, src_data_loader, epoch=None):
    """Select high confidence samples from target domain."""

    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # compute embeddings for labeled dataset
    labeled_reviews = []
    labeled_labels = []
    for (reviews, mask, segment, labels, indices) in src_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        with torch.no_grad():
            embeddings = encoder(reviews, mask, segment)
        labeled_reviews.append(embeddings.cpu())
        labeled_labels.append(labels.cpu())
    labeled_reviews = torch.cat(labeled_reviews, dim=0)
    labeled_labels = torch.cat(labeled_labels, dim=0)

    # compute embeddings for unlabeled dataset, cluster and generate pseudo-labels
    unlabeled_reviews = []
    unlabeled_indices = []
    for (reviews, mask, segment, _, indices) in tgt_data_loader:
        reviews = make_cuda(reviews)
        mask = make_cuda(mask)
        segment = make_cuda(segment)
        with torch.no_grad():
            embeddings = encoder(reviews, mask, segment)
        unlabeled_reviews.append(embeddings.cpu())
        unlabeled_indices.append(indices.cpu())
    unlabeled_reviews = torch.cat(unlabeled_reviews, dim=0)
    unlabeled_indices = torch.cat(unlabeled_indices, dim=0)

    # concatenate labeled and unlabeled embeddings
    all_embeddings = torch.cat([labeled_reviews, unlabeled_reviews], dim=0)

    # compute similarity matrix using cosine similarity
    sim_matrix = torch.mm(all_embeddings, all_embeddings.t()).cpu()
    sim_matrix = (sim_matrix + sim_matrix.t()) / 2.0

    # perform clustering on similarity matrix
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(sim_matrix.numpy())

    # plot embeddings using a scatter plot
    fig, ax = plt.subplots()
    colors = np.array(['r', 'g', 'b', 'y', 'm', 'c'])
    ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=colors[kmeans.labels_])
    plt.show()

    # generate pseudo-labels for unlabeled data
    pseudo_labels = torch.zeros(unlabeled_reviews.shape[0], dtype=torch.long)
    for i in range(args.num_clusters):
        cluster_idx = (kmeans.labels_ == i)
        labeled_idx = cluster_idx[:labeled_reviews.shape[0]]
        if labeled_idx.any():
            # if cluster has labeled data
            label = labeled_labels[labeled_idx][0]

            # assign pseudo-labels to all unlabeled samples in the cluster
            pseudo_labels[cluster_idx[labeled_reviews.shape[0]:]] = label

        # return pseudo-labeled data
    pseudo_labeled_dataset = TensorDataset(unlabeled_reviews, pseudo_labels, unlabeled_indices)
    new_dataset = ConcatDataset([src_data_loader.dataset, pseudo_labeled_dataset])
    pseudo_labeled_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return pseudo_labeled_loader

