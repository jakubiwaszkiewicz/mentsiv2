from random import randint
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torchvision
from torchvision import transforms

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
MODEL_PATH='model.pth'

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

randint = randint(0, 1)

if randint == 0:
  rand_class = "cat"
elif randint == 1:
  rand_class='dog'

img_path = f'dogs_vs_cats/test/{rand_class}s/{rand_class}.18.jpg'


# Load in custom image and convert the tensor values to float32
image = torchvision.io.read_image(str(img_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
image = image / 255.

image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
])

# Transform target image
image_transformed = image_transform(image)

model.eval()
with torch.inference_mode():
    # Add an extra dimension to image
    image_transformed_with_batch_size = image_transformed.unsqueeze(dim=0)

    # Make a prediction on image with an extra dimension
    image_pred = model(image_transformed.unsqueeze(dim=0).to(device))


# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
image_pred_probs = torch.softmax(image_pred, dim=1)

# Convert prediction probabilities -> prediction labels
image_pred_label = torch.argmax(image_pred_probs, dim=1)

class_names = ['cats', 'dogs']

image_pred_class = class_names[image_pred_label] # put pred label to CPU, otherwise will error
print(f"Model is saying: {image_pred_class}")

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(np.squeeze(image.permute(1, 2, 0), axis = 2), cmap='gray') # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Image shape: {image.shape}")
plt.axis(False);
plt.show()

exit()



