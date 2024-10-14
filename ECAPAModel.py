'''
This part is used to train the speaker model and evaluate the performances
'''
import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
# from loss import AAMsoftmax
from loss import *

# from model_ori import ECAPA_TDNN
# from model_1 import ECAPA_TDNN
# from model_21 import ECAPA_TDNN
# from model_22 import ECAPA_TDNN
from model_3 import ECAPA_TDNN

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()

		# for n, p in self.speaker_encoder.named_parameters():
		# 	if "fc6" not in n:
		# 		p.requires_grad = False

		## Classifier
		# self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
		self.speaker_loss    = FCNClassifier(n_class = n_class).cuda()

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()
			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)

			nloss, prec, output       = self.speaker_loss.forward(speaker_embedding, labels)
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%%\r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		labels = []
		embeddings = []
		lines = open(eval_list).read().splitlines()
		for line in lines:
			labels.append(int(line.split()[0]))
			files.append(line.split()[1])
		index_dict = {filename: index for index, filename in enumerate(files)}

		setfiles = list(set(files))
		setfiles.sort(key=lambda x: index_dict[x])
		pred_label = []
		top1, index, loss = 0,0,0
		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
			# data_1 = torch.FloatTensor(numpy.stack([audio],axis=0))

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=1)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda()
			# data_2 = torch.FloatTensor(feats)

			label = torch.tensor([int(labels[idx])]).cuda()

			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)


			# embeddings[file] = [embedding_1, embedding_2]
				# nloss, prec, output       = self.speaker_loss.forward(embedding_1, label)
				nloss, prec_eval, probabilities_eval      = self.speaker_loss.forward(embedding_1, label)
				max_prob, predicted_emotion_eval = torch.max(probabilities_eval, dim=1)
				
			
			# top1 += prec_eval.detach().cpu().numpy()
			# print(embedding_2)
			embeddings.append(embedding_1.tolist())
			top1 += prec_eval
			index += len(label)
			loss += nloss
			pred_label.append(predicted_emotion_eval.item())
			acc = top1/index*len(label)
		
		return loss/(idx), acc, labels, pred_label, files, embeddings

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)