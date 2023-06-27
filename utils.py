import torch
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
from ldm.util import instantiate_from_config

from PIL import Image
import groundingdino.datasets.transforms as T
import torchvision.transforms.functional as F


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# image transformation for groudning DINO
def grounding_transform(image_source):
    transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_transformed, _ = transform(image_source, None)
    return image_transformed

def gligen_transform(image_source, device):
    input_image = F.pil_to_tensor( image_source.resize((512,512)) )
    input_image = ( input_image.float().unsqueeze(0).to(device) / 255 - 0.5 ) / 0.5
    return input_image


def load_ckpt(ckpt_path, device):

    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config
