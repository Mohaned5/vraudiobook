import torch
import os
from transformers import ViTModel, ViTImageProcessor
from utils import text_encoder_forward
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils import latents_to_images, downsampling, merge_and_save_images
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image
from models.celeb_embeddings import embedding_forward
import models.embedding_manager
import importlib

# seed = 42
# set_seed(seed)  
# torch.cuda.set_device(0)

# set your sd2.1 path
model_path = "./stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_path)   
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae = pipe.vae
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

input_dim = 64

experiment_name = "man_GAN"   # "normal_GAN", "man_GAN", "woman_GAN" , 
if experiment_name == "normal_GAN":
    steps = 10000
elif experiment_name == "man_GAN":
    steps = 7000
elif experiment_name == "woman_GAN":
    steps = 6000
else:
    print("Hello, please notice this ^_^")
    assert 0


original_forward = text_encoder.text_model.embeddings.forward
text_encoder.text_model.embeddings.forward = embedding_forward.__get__(text_encoder.text_model.embeddings)
embedding_manager_config = OmegaConf.load("datasets_face/identity_space.yaml")
Embedding_Manager = models.embedding_manager.EmbeddingManagerId_adain(  
        tokenizer,
        text_encoder,
        device = device,
        training = True,
        experiment_name = experiment_name, 
        num_embeds_per_token = embedding_manager_config.model.personalization_config.params.num_embeds_per_token,            
        token_dim = embedding_manager_config.model.personalization_config.params.token_dim,
        mlp_depth = embedding_manager_config.model.personalization_config.params.mlp_depth,
        loss_type = embedding_manager_config.model.personalization_config.params.loss_type,
        vit_out_dim = input_dim,
)
# embedding_path = os.path.join("final", "normal_10.pt")
# Embedding_Manager.load(embedding_path)
text_encoder.text_model.embeddings.forward = original_forward

print("finish init")

# the path of your generated embeddings
test_emb_path = "demo_embeddings/man_66.pt"  # "test_results/normal_GAN/0000/id_embeddings.pt"
test_emb = torch.load(test_emb_path).cuda()
v1_emb = test_emb[:, 0]
v2_emb = test_emb[:, 1]


index = "chosen_index"
save_dir = os.path.join("test_results/" + experiment_name, index)
os.makedirs(save_dir, exist_ok=True)


'''insert into tokenizer & embedding layer'''
tokens = ["v1*", "v2*"]
embeddings = [v1_emb, v2_emb]
# add tokens and get ids
tokenizer.add_tokens(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# resize token embeddings and set new embeddings
text_encoder.resize_token_embeddings(len(tokenizer), pad_to_multiple_of = 8)
for token_id, embedding in zip(token_ids, embeddings):
    text_encoder.get_input_embeddings().weight.data[token_id] = embedding

prompts_list = ["a photo of v1* v2*, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a Superman outfit, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a spacesuit, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a red sweater, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a purple wizard outfit, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a blue hoodie, facing to camera, best quality, ultra high res",
    "v1* v2* wearing headphones, facing to camera, best quality, ultra high res",
    "v1* v2* with red hair, facing to camera, best quality, ultra high res",
    "v1* v2* wearing headphones with red hair, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a Christmas hat, facing to camera, best quality, ultra high res",
    "v1* v2* wearing sunglasses, facing to camera, best quality, ultra high res",
    "v1* v2* wearing sunglasses and necklace, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a blue cap, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a doctoral cap, facing to camera, best quality, ultra high res",
    "v1* v2* with white hair, wearing glasses, facing to camera, best quality, ultra high res",
    "v1* v2* in a helmet and vest riding a motorcycle, facing to camera, best quality, ultra high res",
    "v1* v2* holding a bottle of red wine, facing to camera, best quality, ultra high res",
    "v1* v2* driving a bus in the desert, facing to camera, best quality, ultra high res",
    "v1* v2* playing basketball, facing to camera, best quality, ultra high res",
    "v1* v2* playing the violin, facing to camera, best quality, ultra high res",
    "v1* v2* piloting a spaceship, facing to camera, best quality, ultra high res",
    "v1* v2* riding a horse, facing to camera, best quality, ultra high res",
    "v1* v2* coding in front of a computer, facing to camera, best quality, ultra high res",
    "v1* v2* laughing on the lawn, facing to camera, best quality, ultra high res",
    "v1* v2* frowning at the camera, facing to camera, best quality, ultra high res",
    "v1* v2* happily smiling, looking at the camera, facing to camera, best quality, ultra high res",
    "v1* v2* crying disappointedly, with tears flowing, facing to camera, best quality, ultra high res",
    "v1* v2* wearing sunglasses, facing to camera, best quality, ultra high res",
    "v1* v2* playing the guitar in the view of left side, facing to camera, best quality, ultra high res",
    "v1* v2* holding a bottle of red wine, upper body, facing to camera, best quality, ultra high res",
    "v1* v2* wearing sunglasses and necklace, close-up, in the view of right side, facing to camera, best quality, ultra high res",
    "v1* v2* riding a horse, in the view of the top, facing to camera, best quality, ultra high res",
    "v1* v2* wearing a doctoral cap, upper body, with the left side of the face facing the camera, best quality, ultra high res",
    "v1* v2* crying disappointedly, with tears flowing, with left side of the face facing the camera, best quality, ultra high res",
    "v1* v2* sitting in front of the camera, with a beautiful purple sunset at the beach in the background, best quality, ultra high res",
    "v1* v2* swimming in the pool, facing to camera, best quality, ultra high res",
    "v1* v2* climbing a mountain, facing to camera, best quality, ultra high res",
    "v1* v2* skiing on the snowy mountain, facing to camera, best quality, ultra high res",
    "v1* v2* in the snow, facing to camera, best quality, ultra high res",
    "v1* v2* in space wearing a spacesuit, facing to camera, best quality, ultra high res",
]

for prompt in prompts_list:
    image = pipe(prompt, guidance_scale = 8.5).images[0]
    save_img_path = os.path.join(save_dir, prompt.replace("v1* v2*", "a person") + '.png')
    image.save(save_img_path)
    print(save_img_path)