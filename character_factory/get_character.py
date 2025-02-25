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

experiment_name = "woman_GAN"   # "normal_GAN", "man_GAN", "woman_GAN" , 
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
embedding_path = os.path.join("training_weight", "normal_GAN", "woman3.pt")
Embedding_Manager.load(embedding_path)
text_encoder.text_model.embeddings.forward = original_forward

print("finish init")

# sample a z
random_embedding = torch.randn(1, 1, input_dim).to(device)

# map z to pseudo identity embeddings
_, emb_dict = Embedding_Manager(tokenized_text=None, embedded_text=None, name_batch=None, random_embeddings = random_embedding, timesteps = None,)

test_emb = emb_dict["adained_total_embedding"].to(device)

v1_emb = test_emb[:, 0]
v2_emb = test_emb[:, 1]
embeddings = [v1_emb, v2_emb]

index = "0010"
save_dir = os.path.join("test_results/" + experiment_name, index)
os.makedirs(save_dir, exist_ok=True)
test_emb_path = os.path.join(save_dir, "id_embeddings.pt")
torch.save(test_emb, test_emb_path)

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
    "v1* v2* wearing a blue hoodie, facing to camera, best quality, ultra high res",
]

for prompt in prompts_list:
    image = pipe(prompt, guidance_scale = 8.5).images[0]
    save_img_path = os.path.join(save_dir, prompt.replace("v1* v2*", "a person") + '.png')
    image.save(save_img_path)
    print(save_img_path)
