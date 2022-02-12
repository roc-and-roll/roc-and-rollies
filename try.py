import torch
import torchvision
from PIL import Image
from networks import load_weights
from utils.data_loading import get_transforms

img = Image.open(
    '/home/tom/Code/js/roc-and-rollies/train/d12/d12_4/d12_color615.jpg')
input = get_transforms(256, 3)(img)
batch = torch.unsqueeze(input, dim=0)

model = torchvision.models.resnet18()
network = load_weights(
    model,
    '/home/tom/Code/js/roc-and-rollies/logs/training/training/2022-02-12T16:14:19.982341/checkpoints/000010.pt',
    key='network')
network.eval()

with torch.no_grad():
    output = torch.nn.functional.softmax(network(batch), dim=1)
    print(torch.argmax(output, dim=1))
