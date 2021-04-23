import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import argparse
from tqdm import tqdm
import logging


parser = argparse.ArgumentParser(description="Training regime")
parser.add_argument('--batch_size', help="Batch size", default=128, type=int)
parser.add_argument('--lr', help="Learning rate", default=1e-3, type=float)
parser.add_argument('--cpu', type=bool)
parser.add_argument('--epochs', help="Number of iteration over \
	the full dataset", default=1, type=int)
parser.add_argument('--level', help="Logging output level", 
	default='info', type=str)

levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

class LeNet(nn.Module):
    def __init__(self, num_classes: int=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2,
                                     stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1)
        self.maxpool2 = nn.AvgPool2d(kernel_size=2,
                                     stride=2)
        self.fc1 = nn.Linear(in_features=16*4*4,
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120,
                             out_features=84)
        self.fc3 = nn.Linear(in_features=84,
                             out_features=num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = x.view(-1, self.flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def flatten(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def dataset(batch_size: int=128):
	transform=transforms.Compose(
		[
	        transforms.ToTensor(),
	        transforms.Normalize((0.1307,), (0.3081,))
	    ]
	 )
	train_dataset = MNIST(
		'../data',
		train=True,
		download=True,
		transform=transform
	)
	test_dataset = MNIST(
		'../data',
		train=False,
		download=True,
		transform=transform
	)
	training_generator = DataLoader(train_dataset, batch_size=batch_size)
	test_generator = DataLoader(test_dataset, batch_size=batch_size)
	logger.info("Dataset has processed...")
	return training_generator, test_generator


def trainig_epoch(model, criterion, optimizer, dataloader, device):
	logger.info("Training has started")
	model.train().to(device)
	correct_prediction = 0
	running_loss = 0
	count = 0
	for x_batch, y_batch in dataloader:

		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)

		optimizer.zero_grad()
		preds = model(x_batch)

		predicted_label = torch.argmax(preds, axis=1)
		current_accuracy = (predicted_label == y_batch).sum()
		correct_prediction += current_accuracy


		loss = criterion(preds, y_batch)

		batch_size = len(x_batch)
		count += batch_size

		dataloader.set_postfix(loss=loss.item(),
			accuracy=current_accuracy.item()/ batch_size)

		loss.backward()
		optimizer.step()
		running_loss += loss

	running_loss = running_loss / count
	correct_prediction = correct_prediction / count
	return running_loss, correct_prediction

def validation_epoch(model, criterion, optimizer, dataloader, device):
	logger.info("Validation has started")
	model.eval().to(device)
	correct_prediction = 0
	running_loss = 0
	count = 0
	for x_batch, y_batch in dataloader:
		
		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)
		preds = model(x_batch)

		predicted_label = torch.argmax(preds, axis=1)
		current_accuracy = (predicted_label == y_batch).sum()
		correct_prediction += current_accuracy

		loss = criterion(preds, y_batch)
		running_loss += loss
		batch_size = len(x_batch)
		count += batch_size

		dataloader.set_postfix(loss=loss.item(),
			accuracy=current_accuracy.item() / batch_size)
		
	running_loss = running_loss / count
	correct_prediction = correct_prediction / count
	return running_loss, correct_prediction


def train(model, epochs, criterion, optimizer, dataloader, device):
	training_dataloader, test_dataloader = dataset(128)
	for epoch in range(epochs):
		logger.info(f"Epoch={epoch+1}")
		report = "Epoch {} | Training Loss: {} | Training Accuracy: {}"
		with tqdm(training_dataloader, unit='batch', desc='Training') as tqdm_epoch:
			loss, accuracy = trainig_epoch(
				model=model,
				criterion=criterion,
				optimizer=optimizer,
				dataloader=tqdm_epoch,
				device=device
			)
			print(report.format(epoch+1, loss, accuracy))

		report = "Epoch {} | Validatoin Loss: {} | Validation Accuracy: {}"
		with tqdm(test_dataloader, unit='batch', desc='Validation') as tqdm_epoch:
			loss, accuracy = validation_epoch(
				model=model,
				criterion=criterion,
				optimizer=optimizer,
				dataloader=tqdm_epoch,
				device=device
			)
			print(report.format(epoch+1, loss, accuracy))


if __name__=='__main__':
	args = parser.parse_args()

	logger = logging.getLogger(__name__)
	FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s ] %(message)s"
	logging.basicConfig(format=FORMAT)

	level = args.level

	if level not in levels.keys():
		raise ValueError(
			f"Log value is given: {level}"
			f"Available options are: {' | '.join(levels.keys())}"
		)
	
	logger.setLevel(levels[level])

	net = LeNet()
	logger.info("Model is initialized")
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=args.lr)
	if args.cpu is not None:
		device = args.cpu
	else:
		device = torch.device(
			"cuda:0" if torch.cuda.is_available() else "cpu"
		)
	logger.info(f"Model is working on {device}")
	dataloader = dataset(batch_size=args.batch_size)
	train(model=net,
		epochs=args.epochs,
		criterion=criterion,
		optimizer=optimizer,
		dataloader=dataloader,
		device=device
	)