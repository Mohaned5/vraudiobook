# realesrgan_utils.py

from basicsr.archs.rrdbnet_arch import RRDBNet
from .realesrgan import RealESRGANer
import os

def initialize_realesrgan(model_name='RealESRGAN_x4plus_anime_6B', tile=0, gpu_id=None):
    if model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        # Provide the correct path to your weights; adjust as needed.
        model_path = os.path.join('weights', 'RealESRGAN_x4plus_anime_6B.pth')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        gpu_id=gpu_id,
        tile_pad=10,
        pre_pad=0,
        half=True  # assuming you want fp16 precision
    )
    return upsampler

def upscale_image(image, upsampler, outscale=4):
    # Enhance the image using Real-ESRGAN
    output, _ = upsampler.enhance(image, outscale=outscale)
    return output
