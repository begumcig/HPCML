import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import time

optimizer_options = ['sgd', 'sgdn', 'adagrad', 'adadelta', 'adam']
IMAGE_SIZE = 32
NUM_CLASSES = 17



parser = argparse.ArgumentParser(description='HPCML LAB2')
parser.add_argument('--root', type=str, default='/scratch/gd66/spring2019/lab2/kaggleamazon/', metavar='root',
                    help="folder where data is located. default is %(default)s")
parser.add_argument('--batch-size', type=int, default=250, metavar='BATCH_SIZE',
                    help='input batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: %(default)s)')
parser.add_argument('--enable-cuda', action='store_true',
                    help='enable CUDA')
parser.add_argument('--optimizer', '-o', metavar='optimizer', default='sgd',
                    choices=optimizer_options,
                    help='optimizer options: ' +
                        ' | '.join(optimizer_options) +
                        ' (default: %(default)s)')
parser.add_argument('--lr', type=float, default=0.01, metavar='lr',
                    help='learning rate (default: %(default)s)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='momentum',
                    help='SGD momentum (default: %(default)s)')
parser.add_argument('--weight-decay', type=float, default=0.0, metavar='decay',
					help='weight-decay (default: %(default)s)')
parser.add_argument('--num-workers', type=int, default=1, metavar='worker',
					help='num-workers (default: %(default)s)')


args = parser.parse_args()
args.device = None
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')



class DatasetInitializer(Dataset):
	def __init__(self, root, transform=None):
		data = pd.read_csv(root + "train.csv")
		self.img_names = data.iloc[:,0]
		labels_raw = data.iloc[:,1]
		labels = []
		for unified_label_string in labels_raw:
			label_string_list = unified_label_string.split(" ")
			label_array = [0] * 17
			for label in label_string_list:
				label_array[int(label)] = 1
			labels.append(label_array)
		self.labels = torch.FloatTensor(labels)
		self.transform = transform
		self.path = root


	def __len__(self):
		return self.img_names.shape[0]

	def __getitem__(self, idx):
		label = self.labels[idx]
		img_name = self.img_names[idx]
		img = Image.open(self.path + "train-jpg/" + img_name + ".jpg")
		img = img.convert('RGB')
		if self.transform:
			img = self.transform(img)
		sample = {'image': img, 'label':label}
		return sample



class SimplifiedInception(nn.Module):
	def __init__(self, in_channels):
		super(SimplifiedInception, self).__init__()
		self.conv1_1x1 = nn.Conv2d(in_channels,10,kernel_size=1, stride=1, padding=0)
		self.conv1_3x3 = nn.Conv2d(in_channels,10, kernel_size=3, stride=1, padding=1)
		self.conv1_5x5 = nn.Conv2d(in_channels,10, kernel_size=5, stride=1, padding=2)

	def forward(self, x):
		branch1x1 = self.conv1_1x1(x)
		branch3x3 = self.conv1_3x3(x)
		branch5x5 = self.conv1_5x5(x)
		outputs = [branch1x1, branch3x3, branch5x5]
		return torch.cat(outputs,1)

class Net(nn.Module):
	def  __init__(self):
		super(Net, self).__init__()
		self.Inception1 = SimplifiedInception(3)
		self.Inception2 = SimplifiedInception(30)
		self.fc1 = nn.Linear(8 * 8 * 30, 256)
		self.fc2 = nn.Linear(256, 17)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.Inception1(x),2))
		#print(np.asarray(x.detach()).shape)
		x =  F.relu(F.max_pool2d(self.Inception2(x),2))
		x = x.view(-1, 8 * 8 * 30)
		x = F.relu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x


def precision(k, output, target):
    topk = output.detach().topk(k)[1]
    precision_sum = 0.0
    for i in range(len(output)):
        true_labels = 0
        for j in range(k):
            if int(target[i][int(topk[i][j])]) == 1:
                true_labels += 1
        precision_sum += (float(true_labels)/float(k))
    return precision_sum



def main():
	transform = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor()])
	model = Net().to(args.device)
	dataset = DatasetInitializer(args.root, transform)
	data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
	if(args.optimizer == 'sgd'):
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
	elif(args.optimizer == 'adam'):
		optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay= args.weight_decay )
	elif(args.optimizer == 'sgdn'):
		optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov= True)
	elif(args.optimizer == 'adagrad'):
		optimizer = optim.Adagrad(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
	elif(args.optimizer == 'adadelta'):
		optimizer = optim.Adadelta(model.parameters(), lr= args.lr, weight_decay= args.weight_decay)

	lossf = nn.BCELoss()
	epoch_time = []
	batch_time = []
	loader_time = []
	loss_mean = []
	precision_1_list = []
	precision_3_list = []

	for i in range(1, args.epochs+1):
		epoch_start = time.monotonic()
		model.train()
		runing_loss = 0.0
		batch_start = time.monotonic()
		data_loader_start = time.monotonic()
		#epoch_batch_time = []
		#epoch_data_loading_time = [] 
		#uncomment these if you'd like to get average data loading and batch execution time for every epoch. 
		#the verbose outputs only prints batch execution and data loading times for every batch and total. 
		precision_1 = 0.0
		precision_3 = 0.0
		for batch_idx, sample in enumerate(data_loader):
			data_loader_end = time.monotonic()
			img, label = sample['image'], sample['label']
			img = img.to(args.device)
			label = label.to(args.device)
			optimizer.zero_grad()
			output = model(img)
			loss = lossf(output, label)
			runing_loss += loss.item()
			loss.backward()
			optimizer.step()
			batch_end = time.monotonic()
			precision_1 += precision(1, output, label)
			precision_3 += precision(3, output, label)
			''' for verbose outputs
			print("----Batch stats----")
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tPrecision@1: {:.6f} \tPrecision@3: {:.6f}'.format(
	            i, batch_idx * len(img), len(data_loader.dataset),
	            100. * batch_idx / len(data_loader), loss.data.item(), precision_1/len(output), precision_3/len(output) ))
			print('Batch time: {:6f} \t Data Loader time: {:.6f}'.format(batch_end - batch_start, data_loader_end - data_loader_start ))
			print("-------------------")'''
			
			batch_time.append(batch_end - batch_start)
			#epoch_batch_time.append(batch_end - batch_start)
			loader_time.append(data_loader_end - data_loader_start)
			#epoch_data_loading_time.append(data_loader_end - data_loader_start)
			batch_start = time.monotonic()
			data_loader_start = time.monotonic()
		epoch_end  = time.monotonic()
		epoch_time.append(epoch_end - epoch_start)
		runing_loss_mean = (runing_loss) / len(data_loader)
		loss_mean.append(runing_loss_mean)
		precision_1_mean = float(precision_1) / len(data_loader.dataset)
		precision_3_mean = float(precision_3) / len(data_loader.dataset)
		precision_1_list.append(precision_1_mean)
		precision_3_list.append(precision_3_mean)
		''' for verbose outputs
		print("----Epoch stats----")
		print('Epoch:{} \t Epoch time: {:.6f} \t Epoch mean loss: {:.6f} \t Epoch mean precision @1: {:.6f} \t Epoch mean precision @3: {:.6f} '.format(
			i, epoch_end- epoch_start, runing_loss_mean, precision_1_mean, precision_3_mean))
		print("-------------------")'''
		
		
	total_epoch_time = 	float(sum(epoch_time))
	avg_epoch_time = total_epoch_time / len(epoch_time)
	total_batch_time = float(sum(batch_time))
	avg_batch_time = total_batch_time / len(batch_time)
	total_loading_time = float(sum(loader_time))
	avg_loading_time = total_loading_time / len(loader_time)
	avg_loss = float(sum(loss_mean)) / float(len(loss_mean)) 
	avg_precision_1 = float(sum(precision_1_list)) / float(len(precision_1_list))
	avg_precision_3 = float(sum(precision_3_list)) / float(len(precision_3_list))

	print('Total time for epochs: {:.6f}'.format(total_epoch_time))
	print('Total time for batches: {:.6f}'.format(total_batch_time))
	print('Total time for data loading: {:.6f}'.format(total_loading_time))
	print('Average time for epochs: {:.6f}'.format(avg_epoch_time))
	print('Average time for batches: {:.6f}'.format(avg_batch_time))
	print('Average time for data loading: {:.6f}'.format(avg_loading_time))
	print('Average loss: {:.6f}'.format(avg_loss))
	print('Average precision@1: {:.6f}'.format(avg_precision_1))
	print('Average precision@3: {:.6f}'.format(avg_precision_3))


if __name__ == "__main__":
	main()
	