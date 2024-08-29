import torch
from NeuralNetwork import NeuralNetwork
from torchvision import datasets
from torchvision.transforms import ToTensor

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

test_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True,
  transform=ToTensor()
)

# Device to run training on
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

model.eval()
success = 0
for element in test_data:
  x, y = element[0], element[1]
  with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    if (predicted == actual): 
      success +=1
    # print(f"Predicted: {predicted}, actual: {actual}")
print(f"Accuracy of model: {success / len(test_data)}")