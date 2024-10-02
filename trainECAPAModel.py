'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
# from pytorchtools import EarlyStopping

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=60,      help='Maximum number of epochs')#80
parser.add_argument('--batch_size', type=int,   default=64,     help='Batch size')#400
parser.add_argument('--n_cpu',      type=int,   default=0,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
# parser.add_argument('--train_list', type=str,   default="./data_list_k10/data_train_6.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
# parser.add_argument('--train_list', type=str,   default="./data_list_5fold/ses2_out_sf_data_train.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="./data/IEMOCAP_full_release",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
# parser.add_argument('--eval_list',  type=str,   default="./data_test_one_sp.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="./data_list_k10/data_test_6.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
# parser.add_argument('--eval_list',  type=str,   default="./data_list_5fold/ses2_out_sf_data_test.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="./data/IEMOCAP_full_release",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
# parser.add_argument('--eval_path',  type=str,   default="./",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="./musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="./RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case')
# parser.add_argument('--save_path',  type=str,   default="./1_ses_out_based/",                                        help='Path to save the score.txt and models')
parser.add_argument('--save_path',  type=str,   default="./emb/80_20_no_fr/FCN/V3_1",                                     help='Path to save the score.txt and models')
# parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V2/v21/exps/exp1/model_0120.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V2/v22/exps/exp1/model_0122.model",                                          help='Path of the initial_model')
parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V3/exps/exp1/model_0129.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V1/exps/vox1-O/model_0105.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V1/exps/vox1-H/model_0130.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./ECAPA_variations/V1/exps/vox1-E/model_0125.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./80_20_k10/k1_aug_FCN/V3_model_0129/model/model_6_0011.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./1_ses_out/ses2/k1_aug_FCN/V3_model_0129/model/model_1_0010.model",                                          help='Path of the initial_model')
# parser.add_argument('--initial_model',  type=str,   default="./1_ses_out_based/emb2/FCN/V3/model/model_2_0001.model",                                          help='Path of the initial_model')
## Model and Loss settings
# parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder') #ori
parser.add_argument('--C',       type=int,   default=1008,   help='Channel size for the speaker encoder') #v3
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
# parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser.add_argument('--n_class', type=int,   default=4,   help='Number of emo')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	loss, acc_score, true_label, pred_label, file_name, embeddings = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)

	with open('V3_FCN_model6_emb_1_fr.csv', 'w') as output:
		output.writelines('true_label,pred_label,file_name,embeddings\n')
		for idx in range(len(pred_label)):
			output.writelines(str(true_label[idx]) + ',' + str(pred_label[idx]) + ',' + str(file_name[idx]) + ',' + str(embeddings[idx]) + '\n')
	# print(true_label)
	# print(pred_label)
	print("ACC %2.2f%%"%(acc_score))
	target_names = ['class 0', 'class 1', 'class 2', 'class 3']
	print(classification_report(true_label, pred_label, target_names=target_names))
        
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

# k-10
train_setlist = ['data_train_1.txt', 'data_train_2.txt', 'data_train_3.txt', 'data_train_4.txt', 'data_train_5.txt', 
			  'data_train_6.txt', 'data_train_7.txt', 'data_train_8.txt', 'data_train_9.txt', 'data_train_10.txt']
eval_setlist = ['data_val_1.txt', 'data_val_2.txt', 'data_val_3.txt', 'data_val_4.txt', 'data_val_5.txt', 
			 'data_val_6.txt', 'data_val_7.txt', 'data_val_8.txt', 'data_val_9.txt', 'data_val_10.txt']

# for train_list in train_setlist:

# early_stopping = EarlyStopping(patience=15, verbose=True)

# session out
# train_setlist = ["ses1_out_sf_data_train.txt", "ses2_out_sf_data_train.txt", "ses3_out_sf_data_train.txt","ses4_out_sf_data_train.txt", "ses5_out_sf_data_train.txt"]
# eval_setlist = ["ses1_out_sf_data_val.txt", "ses2_out_sf_data_val.txt", "ses3_out_sf_data_val.txt", "ses4_out_sf_data_val.txt", "ses5_out_sf_data_val.txt"]

for i in range(len(train_setlist)):
	train_list_path = "./data_list_k10/" + train_setlist[i]
	eval_list_path = "./data_list_k10/" + eval_setlist[i]
	# parser.add_argument('--train_list', type=str, default= train_list)
	# parser.add_argument('--eval_list', type=str, default= eval_list)
	print("train list: ", train_list_path)
	print("eval list: ", eval_list_path)
	## Define the data loader
	trainloader = train_loader(train_list=train_list_path, **vars(args))
	trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

	EERs = []
	ACC_list = []
	# train_ACC_list = []
	# error_list = []
	loss_list = []
	score_file = open(args.score_save_path, "a+")

	while(1):
		## Training for one epoch
		loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)
		# train_ACC_list.append(acc)
		## Evaluation every [test_step] epochs
		if epoch % args.test_step == 0:
			s.save_parameters(args.model_save_path + "/model_%01d_%04d.model"%(i+1, epoch))
			eval_loss, eval_acc, _, _, _, _ = s.eval_network(eval_list = eval_list_path, eval_path = args.eval_path)
			ACC_list.append(eval_acc)
			loss_list.append(eval_loss)

			print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, train_ACC %2.2f%%, val_ACC %2.2f%%, best_val_ACC %2.2f%%, val_loss %2.2f, min_loss %2.2f"%(epoch, acc, eval_acc, max(ACC_list), eval_loss, min(loss_list)))
			score_file.write("%d epoch, LR %f, LOSS %f, train_ACC %2.2f%%, val_ACC %2.2f%%, best_val_ACC %2.2f%%, val_loss %2.2f, min_loss %2.2f\n"%(epoch, lr, loss, acc, eval_acc, max(ACC_list), eval_loss, min(loss_list)))
			score_file.flush()

		# if early_stopping.early_stop:
		# 	print("Early Stopping")
		# 	epoch = 1
		# 	s = ECAPAModel(**vars(args))
		# 	s.load_parameters(args.initial_model)
		# 	break
		if epoch >= args.max_epoch:
			# quit()
			epoch = 1
			##load model from initial state
			s = ECAPAModel(**vars(args))
			s.load_parameters(args.initial_model)
			break

		epoch += 1


## run: python trainECAPAModel.py --eval --initial_model exps/pretrain.model
	# python trainECAPAModel.py --initial_model exps/pretrain.model    
	# "args": ["--initial_model", "ECAPA_variations/V1/exps/vox1-O/model_0105.model"]
