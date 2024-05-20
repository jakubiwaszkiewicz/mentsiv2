import torch
from torch import nn

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"



train_dir = "dogs_vs_cats/train"
test_dir = "dogs_vs_cats/test"

import random
import glob

random.seed(713)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# Write transform for image
data_transform = transforms.Compose([
  transforms.Resize(size=IMAGE_SIZE), # Resize the images to IMAGE_SIZE
  transforms.RandomHorizontalFlip(p=0.5), # Flip the images randomly on the horizontal, p = probability of flip, 0.5 = 50% chance=
  transforms.ToTensor() # Turn the image into a torch.Tensor and converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

from torchvision import datasets


from torch.utils.data import DataLoader


# Note that batch size will now be 1.

# Set image size.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# Create training transform with TrivialAugment
train_transform = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.TrivialAugmentWide(),
  transforms.ToTensor()])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.ToTensor()])


# Turn image folders into Datasets
train_data_augmented_all = datasets.ImageFolder(train_dir, transform=train_transform)
test_data_augmented_all = datasets.ImageFolder(test_dir, transform=test_transform)

# Setting dataset parameters
# Set the number of train images to use, max = 20000
train_num_images = 20000
# Set the number of test images to use, max = 5000
test_num_images = 2000
# Size of batch
BATCH_SIZE = 32
# rand
torch.manual_seed(713)

# Create a shuffled list of train images
train_img_list = torch.randperm(len(train_data_augmented_all)).tolist()
train_subset = train_img_list[:train_num_images]

# Create a shuffled list of test images
test_img_list = torch.randperm(len(test_data_augmented_all)).tolist()
test_subset = test_img_list[:test_num_images]

# Create the train subset dataset
train_data_subset = Subset(train_data_augmented_all, train_subset)

# Create the test subset dataset
test_data_subset = Subset(test_data_augmented_all, test_subset)



train_dataloader_augmented = DataLoader(train_data_subset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader_augmented = DataLoader(test_data_subset, batch_size=BATCH_SIZE, shuffle=False)


# Creating a CNN-based image classifier.
class ConvolutionalNeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(64),
      nn.MaxPool2d(2))
    self.conv_layer_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(256),
      nn.MaxPool2d(2))
    self.conv_layer_3 = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(512),
      nn.MaxPool2d(2))
    self.conv_layer_4 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(512),
      nn.MaxPool2d(2))
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=512*3*3, out_features=2))
  def forward(self, x: torch.Tensor):
    x = self.conv_layer_1(x)
    x = self.conv_layer_2(x)
    x = self.conv_layer_3(x)
    x = self.conv_layer_4(x)
    x = self.conv_layer_4(x)
    x = self.conv_layer_4(x)
    x = self.classifier(x)
    return x
# Instantiate an object.
model = ConvolutionalNeuralNetwork().to(device)

from math import ceil

def train_step(
    step_turn: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer):
  # Put model in train mode
  model.train()
  print(f" Epoch: {step_turn} | Train step")
  step_turn += 1
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  # Loop through data loader data batches
  for batch_id, (X, y) in tqdm(enumerate(dataloader), total=ceil(train_num_images/BATCH_SIZE)):
    # Send data to target device
    X, y = X.to(device), y.to(device)
    # 1. Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    # Calculate and accumulate accuracy metric across all batches
    y_soft_max = torch.softmax(y_pred, dim=1)
    y_pred_class = torch.argmax(y_soft_max, dim=1)

    train_acc = train_acc + (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(
    step_turn: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module):
  # Put model in eval mode
  model.eval()
  print(f" Epoch: {step_turn} | Test step")
  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
    for batch_id, (X, y) in tqdm(enumerate(dataloader), total=ceil(test_num_images/BATCH_SIZE)):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      test_pred_logits = model(X)

      # 2. Calculate and accumulate loss
      loss = loss_fn(test_pred_logits, y)
      test_loss += loss.item()

      # Calculate and accumulate accuracy
      test_pred_labels = test_pred_logits.argmax(dim=1)
      test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc



from tqdm.auto import tqdm
# 1. Take in various parameters required for training and test steps
def analysis(
  model: torch.nn.Module,
  train_dataloader: torch.utils.data.DataLoader,
  test_dataloader: torch.utils.data.DataLoader,
  optimizer: torch.optim.Optimizer,
  epochs: int,
  loss_fn: torch.nn.Module,
  ):
  # 2. Create empty results dict
  results = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
  }
  # adding epochs count to see how fast is the CNN
  step_turn = 1
  # 3. Loop through training and testing steps for a number of epochs
  for epoch in range(epochs):
    train_loss, train_acc = train_step(step_turn=step_turn, model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
    test_loss, test_acc = test_step(step_turn=step_turn, model=model, dataloader=test_dataloader, loss_fn=loss_fn)

    # 4. Print out what's happening
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.3f} | "
        f"train_acc: {train_acc:.2f} | "
        f"test_loss: {test_loss:.3f} | "
        f"test_acc: {test_acc:.2f}"
    )

    # 5. Update results dict
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  # 6. Return the filled results at the end of the epochs
  return results



# Set random seeds
torch.manual_seed(713)
torch.cuda.manual_seed(713)

# Set number of epochs
NUM_EPOCHS = 3

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

# Start the timer
from timeit import default_timer as timer

print("Train has been started...")
start_time = timer()

# Train model_0
model_results = analysis(
  model=model,
  train_dataloader=train_dataloader_augmented,
  test_dataloader=test_dataloader_augmented,
  optimizer=optimizer,
  epochs=NUM_EPOCHS,
  loss_fn=loss_fn,
  )

# End the timer and print out how long it took
end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")


torch.save(model.state_dict(), 'model.pth')

print("\"model.pth\" has been created...")
print("To run the tests, run the predict_model.py.")

print("Exiting program...")

