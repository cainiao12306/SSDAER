"""Main script for Maximum Mean Discrepancy (MMD)."""
import sys
sys.path.append("../")
import param
from train.adapt_mmd import warmtrain,evaluate,train
from modules.extractor import BertEncoder
from modules.matcher import BertClassifier
from utils import CSV2Array, convert_examples_to_features, get_data_loader, init_model,select_credible_samples
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import os
import random
import argparse

def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="b2",help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="fz",help="Specify tgt dataset")

    parser.add_argument('--pretrain',default=False, action='store_true',
                        help='Force to pretrain source encoder/classifier')

    parser.add_argument('--adapt',default=False, action='store_true',
                        help='Force to adapt target encoder')

    parser.add_argument('--seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--train_seed', type=int, default=42,
                        help="Specify random state")

    parser.add_argument('--load', default=False, action='store_true',
                        help="Load saved model")

    parser.add_argument('--model', type=str, default="bert",
                        choices=["bert"],
                        help="Specify model type")

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Specify maximum sequence length")

    parser.add_argument('--alpha', type=float, default=1.0,
                        help="cls loss weight")

    parser.add_argument('--beta', type=float, default=0.1,
                        help="mmd loss weight")

    parser.add_argument('--temperature', type=int, default=20,
                        help="Specify temperature")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--clip_value", type=float, default=0.01,
                        help="lower and upper clip value for disc. weights")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Specify the number of epochs for adaptation")

    parser.add_argument('--log_step', type=int, default=50,
                        help="Specify log step size for adaptation")

    parser.add_argument('--source_only', type=int, default=0,
                        help="Specify log step size for adaptation")

    parser.add_argument('--num_mc_samples', type=int, default=10,
                        help="number of dropout")

    parser.add_argument('--num_iterations', type=int, default=2,
                        help="number of src_data_loader iterations")

    parser.add_argument('--confidence_threshold', type=float, default=0.99,
                        help="None")

    parser.add_argument('--uncertainty_threshold', type=float, default=0.01,
                        help="None")

    parser.add_argument('--reg1', type=float, default=1.0,
                        help='Hyperparam for loss')

    parser.add_argument('--reg2', type=float, default=1.0,
                        help='Hyperparam for loss')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)#为random设置种子
    torch.manual_seed(seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
    if torch.cuda.device_count() > 0:#如果有GPU
        torch.cuda.manual_seed_all(seed)#为所有的GPU设置种子


new_args = {'src':'watches','tgt':'shoes','pretrain':False,'adapt':False,'seed':42,'train_seed':42,'load':False,'model':'bert',
            'max_seq_length':128,'alpha':1.0,'beta':0.1,'temperature':20,'max_grad_norm':1.0,'clip_value':0.01,'batch_size':32,
            'num_epochs':20,'log_step':50,'source_only':0,'num_mc_samples':10,'num_iterations':2,'confidence_threshold':0.9,
            'uncertainty_threshold':0.1,'reg1':1,'reg2':1}

def main():
    args = parse_arguments()
    args.__dict__.update(new_args)
    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seed: " + str(args.seed))
    print("train_seed: " + str(args.train_seed))
    print("model_type: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    print("num_epochs: " + str(args.num_epochs))
    print("cls loss weight: " + str(args.alpha))
    print("mmd loss weight: " + str(args.beta))
    print("temperature: " + str(args.temperature))
    set_seed(args.train_seed)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained('../bert-base-multilingual-cased')

    # preprocess data
    print("=== Processing datasets ===")
 
    src_x, src_y = CSV2Array(os.path.join('../data', args.src, args.src + '.csv'))
    tgt_x, tgt_y = CSV2Array(os.path.join('../data', args.tgt, args.tgt + '.csv'))    
    tgt_train_x, tgt_train_y = tgt_x, tgt_y
    tgt_train_x, tgt_valid_x, tgt_train_y, tgt_valid_y = train_test_split(tgt_x, tgt_y,
                                                                        test_size=0.1,
                                                                        stratify=tgt_y,#按tgt_y中的标签（0，1）比例进行分类
                                                                        random_state=args.seed)

    #convert_examples_to_features 将InputExample类转化为InputFeatures类
    #【InputExample】guid: 示例的唯一id；words: 示例句子。；label: 示例的标签。
    #【InputFeatures】input_ids 、attention_mask、token_type_ids在加一个label_id

    src_features = convert_examples_to_features(src_x, src_y, args.max_seq_length, tokenizer)
    tgt_features = convert_examples_to_features(tgt_train_x, tgt_train_y, args.max_seq_length, tokenizer)
    tgt_valid_features = convert_examples_to_features(tgt_valid_x, tgt_valid_y, args.max_seq_length, tokenizer)

    # load dataset
    # 使用get_data_loader()函数加载训练集、验证集和测试集的数据，用来PyTorch的数据读取
    # 三个部分：train_dataset返回一个batch的所有数据，collate_fn将所有数据封装到一起，后面就可以用data_loader得到每一个batch的数据

    src_data_loader = get_data_loader(src_features, args.batch_size,"train")
    tgt_data_train_loader = get_data_loader(tgt_features, args.batch_size,"train")
    tgt_data_valid_loader = get_data_loader(tgt_valid_features, args.batch_size,"dev")
    # load models
    if args.model == 'bert':
        src_encoder = BertEncoder()
        src_classifier = BertClassifier()

    if args.load:
        src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path+'mmdbestmodel')
        src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path+'mmdbestmodel')
    else:
        src_encoder = init_model(args, src_encoder)
        src_classifier = init_model(args, src_classifier)

    best_f1 = 0.0
    print("=== Warmup training for source domain")
    warm_src_encoder, warm_src_classifier, best_f1 = warmtrain(args, src_encoder, src_classifier, src_data_loader,tgt_data_train_loader,tgt_data_valid_loader)
    print("=== Result of Warmup: ===")
    print(best_f1)
    print("=== SSL training for source domain===")
    src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path + 'mmdbestmodel')
    src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path + 'mmdbestmodel')
    args.num_epochs = 50
    #warm_src_encoder, warm_src_classifier, best_f1 = warmtrain(args, src_encoder, src_classifier, src_data_loader,tgt_data_train_loader, tgt_data_valid_loader)
    pseudo_data_loader = select_credible_samples(args,src_encoder,src_classifier,tgt_data_train_loader)
    src_encoder, src_classifier, best_f1 = train(args,src_encoder,src_classifier,src_data_loader,tgt_data_train_loader,tgt_data_valid_loader,pseudo_data_loader)
    print("=== Result of SSL: ===")
    print(best_f1)

    """src_encoder = init_model(args, src_encoder, restore=param.src_encoder_path + 'mmdbestmodel')
    src_classifier = init_model(args, src_classifier, restore=param.src_classifier_path + 'mmdbestmodel')
    f1_train, eloss_train = evaluate(args, src_encoder, src_classifier, tgt_data_train_loader, src_data_loader, pattern=1000)
    print(f1_train)"""

if __name__ == '__main__':
    main()

