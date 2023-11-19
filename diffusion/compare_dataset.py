import os
import torch
import torch_fidelity
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from knockknock import discord_sender

from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance


class DataLoader:
    def load_real_images(self, path):
        real_images: str = path
        real_images_path: str = sorted([os.path.join(real_images, x) for x in os.listdir(real_images)])
        real_images = [np.array(Image.open(path).convert("RGB")) for path in real_images_path]
        
        return real_images

    def load_fake_images(self, path):
        fake_images: str = path
        fake_images_path: str = sorted([os.path.join(fake_images, x) for x in os.listdir(fake_images)])
        fake_images = [np.array(Image.open(path).convert("RGB")) for path in fake_images_path]
        
        return fake_images

    def preprocess_real_image(self, image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        image = F.center_crop(image, (128, 128))
        return image

    def preprocess_fake_images(self, image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
        # fake_images = F.center_crop(fake_images, (256, 256))
        return image


class MetricsHandler:
    def compute_fid_score(self, real_images, fake_images):
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        return fid.compute()    
    
    def compute_kernel_interception_score(self, real_images, fake_images):
        kid = KernelInceptionDistance(subset_size=25)
        kid.update(real_images, real=True)
        kid.update(fake_images, real=False)
        
        return kid.compute()
        

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_images_path", type=str, default="/home/tmp/scholuka/NCT-CRC-HE-100K/TUM")
    parser.add_argument("--fake_images_path", type=str, default="/vol/fob-vol5/mi22/scholuka/repositorys/medical-image-synthetizing/data/fake_images/tum")
    parser.add_argument("--metric", type=str, default="fid")
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    dataloder: DataLoader = DataLoader()
    metrics_handler: MetricsHandler = MetricsHandler()
    
    args = parse_arguments()

    real_images = dataloder.load_real_images(path="/home/tmp/scholuka/NCT-CRC-HE-100K/TUM")
    fake_images = dataloder.load_fake_images(path="/vol/fob-vol5/mi22/scholuka/repositorys/medical-image-synthetizing_old/data/fake_images/tum")

    real_images = torch.cat([dataloder.preprocess_real_image(image) for image in real_images])
    fake_images = torch.cat([dataloder.preprocess_fake_images(image) for image in fake_images])
    
    if args.metric == "fid":
        print("Computing FID...")
        metric = metrics_handler.compute_fid_score(real_images, fake_images)
        print(float(metric))
    elif args.metric == "kid":
        print("Computing KID...")
        mean, std = metrics_handler.compute_kernel_interception_score(real_images, fake_images)
        print(f"mean: {mean}, std: {std}")
