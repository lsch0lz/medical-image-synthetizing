import os

from PIL import Image
from knockknock import discord_sender
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMPipeline


pipeline = DDPMPipeline.from_pretrained(pretrained_model_name_or_path="/vol/fob-vol5/mi22/scholuka/repositorys/medical-image-synthetizing/TUM/")
pipeline.to("cuda")

webhook_url = "https://discord.com/api/webhooks/1175132168679333908/to8VLfO6Sc9mGmZ_YrD8Hr0bdJM1C6-AQmlFY1aHEprpCnMGIkIBB6E3j4fpmIzS0UUX"
@discord_sender(webhook_url=webhook_url)
def create_dataset():
    for i in tqdm(range(100)):
        image = pipeline().images[0]
        image.save(f"/vol/fob-vol5/mi22/scholuka/repositorys/medical-image-synthetizing/data/fake_images/tum/{i}.png")
        

if __name__ == "__main__":
    create_dataset()