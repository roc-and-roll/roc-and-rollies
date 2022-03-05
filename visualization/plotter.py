import torch
from typing import List
from pytorch_training.extensions import ImagePlotter
from PIL import ImageDraw
import torchvision
import os
import global_config
from pytorch_training.extension import Extension
from torchvision import transforms


class DicePlotter(ImagePlotter):
    def __init__(self, input_images: list, networks: list, log_dir: str, *args, plot_to_logger: bool = False, labels: torch.Tensor = None, **kwargs):
        Extension.__init__(self, *args, **kwargs)
        self.input_images = torch.stack(input_images).to(global_config.device)
        self.networks = networks
        self.image_dir = os.path.join(log_dir, 'images')
        self.log_to_logger = plot_to_logger
        os.makedirs(self.image_dir, exist_ok=True)
        self.labels = labels

    def get_predictions(self) -> List[torch.Tensor]:
        predictions = []
        for network in self.networks:
            output = network(self.input_images)

            with torch.no_grad():
                output = torch.nn.functional.softmax(output, dim=1)
                output = torch.argmax(output, dim=1)

            batch = torch.zeros_like(self.input_images)
            for i, (input_image, label, output) in enumerate(zip(self.input_images, self.labels, output)):
                unnormalized = (input_image + 1) / 2
                image = torchvision.transforms.functional.to_pil_image(unnormalized)
                draw = ImageDraw.Draw(image)
                draw.text((10, 10), f"Label: {label + 1} Output: {output + 1}", fill="red")
                batch[i] = torchvision.transforms.functional.to_tensor(image)
            predictions.append(torch.tensor(batch))
        return predictions
