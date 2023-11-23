import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np # data processing
import matplotlib.pyplot as plt # Data visualization
from tqdm import tqdm # Progress bar

import os
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from types import SimpleNamespace

import timm

import random
import datetime

root_dir = 'Images/' # Insert your data here
sub_folders = ['Cheetahs', 'Lions'] # Insert your classes here
labels = [0, 1]

data = []

for s, l in zip(sub_folders, labels):
	for r, d, f in os.walk(root_dir + s):
		for file in f:
			if '.jpg' in file:
				data.append((os.path.join(s, file), l))

df = pd.DataFrame(data, columns=['file_name', 'label'])

sns.countplot(data = df, x = 'label')
plt.savefig('class_distribution.svg')

fig, ax = plt.subplots(2, 3, figsize=(10, 6))

idx = 0
rand_id = np.random.choice(len(df.file_name), 6, replace=False)
for i in range(2):
	for j in range(3):

		label = df.label[rand_id[idx]]
		file_path = os.path.join(root_dir, df.file_name[rand_id[idx]])

		# Read an image with OpenCV
		image = cv2.imread(file_path)

		# Convert the image to RGB color space.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Resize image
		image = cv2.resize(image, (256, 256))

		ax[i, j].imshow(image)
		ax[i, j].set_title(f"Label: {label} ({'Lion' if label == 1 else 'Cheetah'})")
		ax[i, j].axis('off')
		idx = idx + 1

plt.tight_layout()
plt.savefig('sample_images.svg')

train_df, test_df = train_test_split(
	df,
	test_size = 0.1,
	random_state = 42
)

cfg = SimpleNamespace(**{})

cfg.root_dir = root_dir
cfg.image_size = 256

class CustomDataset(Dataset):
	def __init__(self, cfg, df, transform=None, mode='val'):
		self.root_dir = cfg.root_dir
		self.df = df
		self.file_names = df['file_name'].values
		self.labels = df['label'].values

		if transform:
			self.transform = transform
		else:
			self.transform = A.Compose([
				A.Resize(cfg.image_size, cfg.image_size),
				ToTensorV2()
			])

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		# Get file_path and label for index
		label = self.labels[idx]
		file_path = os.path.join(self.root_dir, self.file_names[idx])

		# Read and image with OpenCV
		image = cv2.imread(file_path)

		# Convert the image to RGB color space.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Apply augmentation
		augmented = self.transform(image=image)
		image = augmented['image']

		# Normalize because ToTensorV2() doesn't normalize the image
		image = image/255

		return image, label

cfg.batch_size = 32

example_dataset = CustomDataset(cfg, df)
example_dataloader = DataLoader(
	example_dataset,
	batch_size = cfg.batch_size,
	shuffle = True,
	num_workers = 0,
)

for (image_batch, label_batch) in example_dataloader:
	print(image_batch.shape)
	print(label_batch.shape)
	break

X = df
y = df.label

train_df, valid_df, y_train, y_test = train_test_split(
	X,
	y,
	test_size = 0.2,
	random_state = 42
)

train_dataset = CustomDataset(cfg, train_df)
valid_dataset = CustomDataset(cfg, valid_df)

train_dataloader = DataLoader(
	train_dataset,
	batch_size = cfg.batch_size,
	shuffle = True
)

valid_dataloader = DataLoader(
	valid_dataset,
	batch_size = cfg.batch_size,
	shuffle = False
)

cfg.n_classes = 2
cfg.backbone = 'resnet18'

model = timm.create_model(
	cfg.backbone,
	pretrained = True,
	num_classes = cfg.n_classes
)

X = torch.randn(cfg.batch_size, 3, cfg.image_size, cfg.image_size)
y = model(X)

criterion = nn.CrossEntropyLoss()

cfg.learning_rate = 1e-4

optimizer = torch.optim.Adam(
	model.parameters(),
	lr = cfg.learning_rate,
	weight_decay = 0
)

cfg.lr_min = 1e-5
cfg.epochs = 5

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
	optimizer, 
	T_max = np.ceil(len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs,
	eta_min = cfg.lr_min
)

def calculate_metric(y, y_pred):
	metric = accuracy_score(y, y_pred)
	return metric

cfg.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(dataloader, model, optimizer, scheduler, cfg):
	# Training mode
	model.train()

	# Init lists to store y and y_pred
	final_y = []
	final_y_pred = []
	final_loss = []

	# Iterate over data
	for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
		X = batch[0].to(cfg.device)
		y = batch[1].to(cfg.device)

		# Zero the parameter gradients
		optimizer.zero_grad()

		with torch.set_grad_enabled(True):
			# Forward: Get model outputs
			y_pred = model(X)

			# Forward: Calculate loss
			loss = criterion(y_pred, y)

			# Convert y and y_pred to lists
			y = y.detach().cpu().numpy().tolist()
			y_pred = y_pred.detach().cpu().numpy().tolist()

			# Extend original list
			final_y.extend(y)
			final_y_pred.extend(y_pred)
			final_loss.append(loss.item())

			# Backward: Optimize
			loss.backward()
			optimizer.step()

		scheduler.step()

	# Calculate statistics
	loss = np.mean(final_loss)
	final_y_pred = np.argmax(final_y_pred, axis=1)
	metric = calculate_metric(final_y, final_y_pred)

	return metric, loss

def validate_one_epoch(dataloader, model, cfg):
	# Validation mode
	model.eval()

	# Init lists to store y and y_pred
	final_y = []
	final_y_pred = []
	final_loss = []

	# Iterate over data
	for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
		X = batch[0].to(cfg.device)
		y = batch[1].to(cfg.device)

		with torch.set_grad_enabled(True):
			# Forward: Get model outputs
			y_pred = model(X)

			# Forward: Calculate loss
			loss = criterion(y_pred, y)

			# Convert y and y_pred to lists
			y = y.detach().cpu().numpy().tolist()
			y_pred = y_pred.detach().cpu().numpy().tolist()

			# Extend original list
			final_y.extend(y)
			final_y_pred.extend(y_pred)
			final_loss.append(loss.item())

	# Calculate statistics
	loss = np.mean(final_loss)
	final_y_pred = np.argmax(final_y_pred, axis=1)
	metric = calculate_metric(final_y, final_y_pred)

	return metric, loss

cfg.n_folds = 5

# Create a new column for cross-validation folds
df['kfold'] = -1

# Initialize the kfold class
skf = StratifiedKFold(n_splits=cfg.n_folds)

# Fill the new column
for fold, (train_, val_) in enumerate(skf.split(X = df, y=df.label)):
	df.loc[val_, 'kfold'] = fold

for fold in range(cfg.n_folds):
	train_df = df[df.kfold != fold].reset_index(drop=True)
	valid_df = df[df.kfold == fold].reset_index(drop=True)

transform_soft = A.Compose([
	A.Resize(cfg.image_size, cfg.image_size),
	A.Rotate(p=0.6, limit=[-45,45]),
	A.HorizontalFlip(p = 0.6),
	A.CoarseDropout(max_holes=1, max_height=64, max_width=64, p=0.3),
	ToTensorV2()
])

def set_seed(seed=1234):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)

	# In general seed PyTorch operations
	torch.manual_seed(seed)

	# If you are using CUDA on 1 GPU, seed it
	torch.cuda.manual_seed(seed)

	# If you are using CUDA on more than 1 GPU, seed them all
	torch.cuda.manual_seed_all(cfg.seed)

	# Certain operations in Cudnn are not deterministic, and this line will force them to behave!
	torch.backends.cudnn.deterministic = True

	# Disable the inbuilt cudnn auto-tuner that finds the best algorithm to use for your hardware.
	torch.backends.cudnn.benchmark = False

cfg.seed = 42

def fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader=None):
    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        
        set_seed(cfg.seed + epoch)
        
        acc, loss = train_one_epoch(train_dataloader, model, optimizer, scheduler, cfg)
        
        if valid_dataloader:
            val_acc, val_loss = validate_one_epoch(valid_dataloader, model, cfg)
        
        print(f'Loss: {loss:.4f} Acc: {acc:.4f}')
        acc_list.append(acc)
        loss_list.append(loss)
        
        if valid_dataloader:
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
    
    return acc_list, loss_list, val_acc_list, val_loss_list, model

def visualize_history(acc, loss, val_acc, val_loss):
	fig, ax = plt.subplots(1, 2, figsize=(12,4))

	ax[0].plot(range(len(loss)), loss, color='darkgrey', label = 'train')
	ax[0].plot(range(len(val_loss)), val_loss, color='cornflowerblue', label = 'valid')
	ax[0].set_title('Loss')

	ax[1].plot(range(len(acc)), acc, color='darkgrey', label='train')
	ax[1].plot(range(len(val_acc)), val_acc, color='cornflowerblue', label='valid')
	ax[1].set_title('Metric (Accuracy)')

	for i in range(2):
		ax[i].set_xlabel('Epochs')
		ax[i].legend(loc='upper right')
	plt.savefig('hist-' + str(datetime.datetime.now().strftime('%H%M%S%f')))

for fold in range(cfg.n_folds):
	train_df = df[df.kfold != fold].reset_index(drop=True)
	valid_df = df[df.kfold == fold].reset_index(drop=True)

	train_dataset = CustomDataset(cfg, train_df, transform = transform_soft)
	valid_dataset = CustomDataset(cfg, valid_df)

	train_dataloader = DataLoader(
		train_dataset,
		batch_size = cfg.batch_size,
		shuffle = True,
		num_workers = 0
	)
	valid_dataloader = DataLoader(
		valid_dataset,
		batch_size = cfg.batch_size,
		shuffle = False,
		num_workers =0
	)

	model = timm.create_model(
		cfg.backbone,
		pretrained = True,
		num_classes = cfg.n_classes,
	)
	model = model.to(cfg.device)

	criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(
		model.parameters(),
		lr = cfg.learning_rate,
		weight_decay = 0
	)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, 
		T_max= np.ceil(len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs,
		eta_min=cfg.lr_min
    )

	acc, loss, val_acc, val_loss, model = fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader)

	visualize_history(acc, loss, val_acc, val_loss)

train_df = df.copy()

train_dataset = CustomDataset(cfg, train_df, transform = transform_soft)

train_dataloader = DataLoader(train_dataset, 
                          batch_size = cfg.batch_size, 
                          shuffle = True, 
                          num_workers = 0,
                         )

model = timm.create_model(cfg.backbone, 
                          pretrained = True, 
                          num_classes = cfg.n_classes)

model = model.to(cfg.device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), 
                             lr = cfg.learning_rate, 
                             weight_decay = 0,
                            )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                       T_max= np.ceil(len(train_dataloader.dataset) / cfg.batch_size) * cfg.epochs,
                                                       eta_min=cfg.lr_min)

acc, loss, val_acc, val_loss, model = fit(model, optimizer, scheduler, cfg, train_dataloader)

test_dataset = CustomDataset(cfg, test_df)

test_dataloader = DataLoader(test_dataset, 
                          batch_size = cfg.batch_size, 
                          shuffle = False, 
                          num_workers = 0,
                         )

dataloader = test_dataloader

# Validation mode
model.eval()

final_y = []
final_y_pred = []

# Iterate over data
for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    X = batch[0].to(cfg.device)
    y = batch[1].to(cfg.device)

    with torch.no_grad():
        # Forward: Get model outputs
        y_pred = model(X)

        # Covert y and y_pred to lists
        y =  y.detach().cpu().numpy().tolist()
        y_pred =  y_pred.detach().cpu().numpy().tolist()

        # Extend original list
        final_y.extend(y)
        final_y_pred.extend(y_pred)

# Calculate statistics
final_y_pred_argmax = np.argmax(final_y_pred, axis=1)
metric = calculate_metric(final_y, final_y_pred_argmax)

test_df['prediction'] = final_y_pred_argmax
