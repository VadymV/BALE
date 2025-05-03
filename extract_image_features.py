import os
import shutil

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel


def recreate_directory(directory_path):
    """
    Checks if a directory exists. If it does, removes it. Then creates a new directory.

    Parameters:
    directory_path (str): Path of the directory to manage.
    """
    try:
        if os.path.exists(directory_path):
            # Remove the directory and its contents
            shutil.rmtree(directory_path)
            print(f"Removed existing directory: {directory_path}")

        # Create the directory
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created new directory: {directory_path}")

    except Exception as e:
        print(f"Error while managing directory {directory_path}: {e}")


class ClipVisionModel(torch.nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super().__init__()

        self.model = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14").eval().to(device)
        self.processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.device = device

    def embed_image(self, image):
        outputs = self.processor(images=image, return_tensors="pt").to(
            self.device)
        outputs = self.model(**outputs)
        image_embeds = outputs.pooler_output
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        return image_embeds


clip_extractor = ClipVisionModel()

for dataset in ['aomic', 'nemo']:
    all_images = 'C:\\Users\\vcx763\\PycharmProjects\\Brainy\\data\\{}\\images'.format(
        dataset)
    embeddings_path = 'C:\\Users\\vcx763\\PycharmProjects\\Brainy\\data\\{}\\images\\embeddings'.format(
        dataset)
    recreate_directory(embeddings_path)
    for filename in os.listdir(all_images):
        if filename.endswith('.jpg') or filename.endswith('.JPG'):
            image = Image.open(os.path.join(all_images, filename))
            image_features = clip_extractor.embed_image(image).float()

            output_path = os.path.join(embeddings_path,
                                       filename.split('.')[0] + '.pt')
            torch.save(image_features, output_path)
