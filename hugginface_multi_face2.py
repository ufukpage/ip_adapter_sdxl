import torch
import cv2
import numpy as np
from diffusers.utils import load_image
from diffusers import AutoPipelineForText2Image, AutoencoderKL, DDIMScheduler, DiffusionPipeline
from insightface.app import FaceAnalysis

image1 = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ai_face2.png")
image2 = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png")

ref_images_embeds = []
ref_unc_images_embeds = []
model_root = "./models/insightface/"  # Replace with your actual model directory path
app = FaceAnalysis(name="buffalo_l", root=model_root, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
for im in [image1, image2]:
    image = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
    faces = app.get(image)
    image = torch.from_numpy(faces[0].normed_embedding)
    image_embeds = image.unsqueeze(0)
    uncond_image_embeds = torch.zeros_like(image_embeds)
    ref_images_embeds.append(image_embeds)
    ref_unc_images_embeds.append(uncond_image_embeds)
ref_images_embeds = torch.stack(ref_images_embeds, dim=0)
ref_unc_images_embeds = torch.stack(ref_unc_images_embeds, dim=0)
single_image_embeds = torch.cat([ref_unc_images_embeds, ref_images_embeds], dim=0).to(dtype=torch.float16)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

base_model_path ="SG161222/RealVisXL_V3.0"
cache_dir = "./model-cache"
pipeline = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    device_map="balanced",
    low_cpu_mem_usage=True,
    cache_dir=cache_dir
)

pipeline.to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID",
                         subfolder=None, 
                         weight_name="ip-adapter-faceid_sdxl.bin", 
                         image_encoder_folder=None)
pipeline.set_ip_adapter_scale(0.7)

pipeline.enable_model_cpu_offload()
generator = torch.Generator(device="cpu").manual_seed(42)

num_images=2

images = pipeline(
    prompt="A photo of a girl wearing a black dress, holding red roses in hand, upper body, behind is the Eiffel Tower",
    ip_adapter_image=[single_image_embeds], guidance_scale=7.5,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", 
    num_inference_steps=30, num_images_per_prompt=2,
    generator=generator
).images