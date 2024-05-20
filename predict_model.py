import random
import seaborn as sns
sns.set_theme()
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from math import ceil
# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
MODEL_PATH='model.pth'
random.seed(713)

loss_fn = CrossEntropyLoss()

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


model = ConvolutionalNeuralNetwork()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

if torch.cuda.is_available():
    model.cuda()

test_dir = "dogs_vs_cats/test"
# Set image size.



# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
  transforms.Resize(IMAGE_SIZE),
  transforms.ToTensor()])


# Turn image folders into Datasets
test_data_augmented_all = datasets.ImageFolder(test_dir, transform=test_transform)

# Setting dataset parameters
# Set the number of test images to use, max = 5000
test_num_images = 500
# Size of batch
BATCH_SIZE = 32
# rand
torch.manual_seed(713)
# Create a shuffled list of test images
test_img_list = torch.randperm(len(test_data_augmented_all)).tolist()
test_subset = test_img_list[:test_num_images]

# Create the test subset dataset
test_data_subset = Subset(test_data_augmented_all, test_subset)
test_dataloader_augmented = DataLoader(test_data_subset, batch_size=BATCH_SIZE, shuffle=False)



def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module):
  # Put model in eval mode
  model.eval()
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

def analysis(
  model: torch.nn.Module,
  test_dataloader: torch.utils.data.DataLoader,
  loss_fn: torch.nn.Module,
  ):
  # 2. Create empty results dict
  results = {
    "test_loss": [],
    "test_acc": []
  }

  test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)


    # 5. Update results dict
  results["test_loss"].append(test_loss)
  results["test_acc"].append(test_acc)
  # 6. Return the filled results at the end of the epochs
  return results

model_results = analysis(
  model=model,
  test_dataloader=test_dataloader_augmented,
  loss_fn=loss_fn,
  )

print(model_results)


