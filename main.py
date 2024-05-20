import argparse
import torch
from pipeline import Pipeline
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description='generating')

# Load data
parser.add_argument('--root_path', type=str, default='./data/price', help='root path of the data files')
parser.add_argument('--tweet_path', type=str, default='./data/tweet', help='root path of the tweet files')
parser.add_argument('--time_length', type=int, default=637, help='length of time stamps')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=8, help='length of input sequence')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--target_size', type=int, default=1, help='dimension of target')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

# Training settings
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--itr', type=int, default=1, help='experiment times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=8e-6, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# master
parser.add_argument('--d_feat', type=int, default=11, help='feature dimension')
parser.add_argument('--d_model', type=int, default=64, help='model dimension')
parser.add_argument('--t_nhead', type=int, default=2, help='number of heads in t-axis')
parser.add_argument('--s_nhead', type=int, default=2, help='number of heads in s-axis')
parser.add_argument('--T_dropout_rate', type=float, default=0.2, help='dropout rate for t-axis')
parser.add_argument('--S_dropout_rate', type=float, default=0.2, help='dropout rate for s-axis')

# finbert
parser.add_argument('--finbert_prefix_len', type=int, default=20, help='Length of the input prefix sequence')
parser.add_argument('--finbert_num_hidden_layers', type=int, default=12, help='Number of hidden layers in the model')
parser.add_argument('--finbert_num_attention_heads', type=int, default=12,
                    help='Number of attention heads in the model')
parser.add_argument('--finbert_hidden_size', type=int, default=768, help='Size of the hidden layers')
parser.add_argument('--finbert_hidden_dropout_prob', type=float, default=0.1,help='Dropout probability for hidden layers')
parser.add_argument('--finbert_prefix_projection', type=int, default=1,help='Whether to use prefix projection (1 for yes, 0 for no)')
parser.add_argument('--finbert_prefix_hidden_size', type=int, default=768,help='Size of the hidden layers for prefix projection')
parser.add_argument('--finbert_max_position_embeddings', type=int, default=512,help='Whether to use return dictionary (1 for yes, 0 for no)')
parser.add_argument('--finbert_use_return_dict', type=int, default=0,help='Whether to use return dictionary (1 for yes, 0 for no)')
parser.add_argument('--temperature', type=float, default=0.8, help='calculate similarity')

parser.add_argument('--alpha_tweet_loss', type=float, default=0.1, help='alpha for tweet')
parser.add_argument('--alpha_data_loss', type=float, default=1, help='alpha for data')
parser.add_argument('--alpha_cat_loss', type=float, default=1, help='alpha for cat')
parser.add_argument('--alpha_align_loss', type=float, default=0.1, help='alpha for align')

# Device
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.pred_len is None:
    args.pred_len = args.seq_len

print('Args in experiment:')
print(args)

print('\n\nStart Running...')
all_mse = []
for ii in range(0, args.itr):
    pipeline = Pipeline(args)
    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    pipeline.train()
    print('>>>>>>>start testing : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    acc, conf_matrix, f1score, auc = pipeline.test()
    print('overall result: acc:{}, conf_matrix:{}, f1score:{}, auc:{}'.format(acc, conf_matrix, f1score, auc))
    torch.cuda.empty_cache()  # 清空CUDA缓存，释放GPU内存
