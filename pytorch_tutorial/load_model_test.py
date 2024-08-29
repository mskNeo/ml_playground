import torch
from torch import nn
from NeuralNetwork import NeuralNetwork

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
