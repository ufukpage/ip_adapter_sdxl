# Prediction interface for Cog for generating two persons
from cog import BasePredictor, Input, Path
import os
import torch
from typing import List
from diffusers.utils import load_image
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers import DiffusionPipeline
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch.nn.functional as F

class TwoPersonPredictor(BasePredictor):
    def setup(self, weights= None) -> None:
        """Load the model into memory"""
        # Initialize base pipeline with explicit configuration
        self.pipeline = DiffusionPipeline.from_pretrained(
            #"stabilityai/stable-diffusion-xl-base-1.0",
            "SG161222/RealVisXL_V5.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        # Set the scheduler to DPM++ SDE Karras
        self.pipeline.scheduler = self.pipeline.scheduler.from_config(
            self.pipeline.scheduler.config,
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            use_karras_sigmas=True
        )

        # Initialize refiner pipeline with explicit configuration
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            vae=self.pipeline.vae,
            text_encoder_2=self.pipeline.text_encoder_2,
        ).to("cuda")

        # Load IP-Adapter with specific configuration
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
            image_encoder_folder="models/image_encoder"
        )
        
        # Set InsightFace model directory
        model_dir = os.path.expanduser('models/insightface')
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize InsightFace with specific model directory
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=model_dir,  # Specify model directory
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Download and prepare the face analysis model
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image):
        """Face detection using InsightFace"""
        # Convert PIL to CV2
        cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.face_analyzer.get(cv2_image)
        boxes = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Add padding
            padding = 30
            height, width = cv2_image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            boxes.append([x1, y1, x2, y2])
        
        return boxes, faces

    def find_closest_face(self, face_crop, face_image1, face_image2):
        """Compare face embeddings using InsightFace"""
        # Convert all images to CV2
        crop_cv2 = cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)
        face1_cv2 = cv2.cvtColor(np.array(face_image1), cv2.COLOR_RGB2BGR)
        face2_cv2 = cv2.cvtColor(np.array(face_image2), cv2.COLOR_RGB2BGR)
        
        # Get face embeddings
        crop_faces = self.face_analyzer.get(crop_cv2)
        face1_faces = self.face_analyzer.get(face1_cv2)
        face2_faces = self.face_analyzer.get(face2_cv2)
        
        if not crop_faces or not face1_faces or not face2_faces:
            # Fallback to first reference face if detection fails
            return face_image1
        
        # Get embeddings
        crop_embedding = torch.tensor(crop_faces[0].embedding)
        face1_embedding = torch.tensor(face1_faces[0].embedding)
        face2_embedding = torch.tensor(face2_faces[0].embedding)
        
        # Calculate cosine similarity
        sim1 = F.cosine_similarity(crop_embedding.unsqueeze(0), face1_embedding.unsqueeze(0))
        sim2 = F.cosine_similarity(crop_embedding.unsqueeze(0), face2_embedding.unsqueeze(0))
        
        # Return the face with higher similarity
        return face_image1 if sim1 > sim2 else face_image2

    def refine_faces(self, image, prompt, face_image1, face_image2, strength=0.4):
        """Refined face detection and matching"""
        boxes, faces = self.detect_faces(image)
        refined_image = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            face_crop = image.crop((x1, y1, x2, y2))
            
            # Find the closest matching input face
            reference_face = self.find_closest_face(face_crop, face_image1, face_image2)
            
            # Refine the face using img2img with IP-Adapter
            refined_face = self.pipeline(
                prompt=prompt,
                image=face_crop,
                ip_adapter_image=reference_face,  # Use IP-Adapter for conditioning
                num_inference_steps=20,
                strength=strength,
                guidance_scale=7.5,
            ).images[0]
            
            # Paste the refined face back
            refined_image.paste(refined_face, (x1, y1))
        
        return refined_image

    def predict(
        self,
        image1: Path = Input(description="Input face image for first person"),
        image2: Path = Input(description="Input face image for second person"),
        prompt: str = Input(
            description="Prompt for generation. Ex: 2 people having dinner in the cafe in the Paris. Bodies and faces towards the camera.",
            default=""
        ),
        negative_prompt: str = Input(
            description="Negative Prompt (applies to both)",
            default="bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth)"
        ),
        scale: float = Input(
            description="Scale (influence of input images)",
            ge=0.0,
            le=1.0,
            default=0.7
        ),
        face_refinement_strength: float = Input(
            description="Strength of face refinement (0.0 to 1.0)",
            ge=0.0,
            le=1.0,
            default=0.4
        ),
        use_face_detailer: bool = Input(
            description="Use face detailer for better face quality",
            default=False
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps",
            ge=1,
            le=500,
            default=40
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize",
            default=None
        ),
        output_width: int = Input(
            description="Output width",
            default=768
        ),
        output_height: int = Input(
            description="Output height",
            default=1024
        ),
        use_refiner: bool = Input(
            description="Use refiner for higher quality (slower)",
            default=False
        )
    ) -> List[Path]:
        """Generate an image with two different persons"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")


        # Load and process face images using load_image instead of Image.open
        face_image1 = load_image(str(image1)).convert("RGB")
        face_image2 = load_image(str(image2)).convert("RGB")

        # Load the predefined masks from huggingface
        mask1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask1.png").convert('L')
        mask2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask2.png").convert('L')
        mask1 = mask1.resize((output_width, output_height))
        mask2 = mask2.resize((output_width, output_height))

        # Process masks to get correct shape [1, num_images_for_ip_adapter, height, width]
        processor = IPAdapterMaskProcessor()
        masks = processor.preprocess([mask1, mask2], height=output_height, width=output_width)
        masks = [masks.reshape(1, 2, output_height, output_width)]

        # Set IP adapter scale
        self.pipeline.set_ip_adapter_scale(scale)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Combine prompts
        prompt = f"{prompt1} {prompt2}"

        # Generate images
        output_paths = []
        for _ in range(num_outputs):
            # Clear VRAM before main generation
            torch.cuda.empty_cache()
            
            # Move models to CPU temporarily if needed
            if use_refiner:
                self.refiner.to("cpu")
            
            # Main generation
            images = self.pipeline(
                prompt=prompt,
                ip_adapter_image=[[face_image1, face_image2]],
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                cross_attention_kwargs={"ip_adapter_masks": masks},
                output_type="latent" if use_refiner else "pil",
                height=output_height,
                width=output_width,
            ).images
            
            if use_refiner:
                # Clear VRAM after main generation
                torch.cuda.empty_cache()
                
                # Move pipeline to CPU and refiner to GPU
                self.pipeline.to("cpu")
                self.refiner.to("cuda")
                
                # Refinement step
                images = self.refiner(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=images,
                    num_inference_steps=20,
                    generator=generator,
                    height=output_height,
                    width=output_width,
                ).images
                
                # Move refiner back to CPU and pipeline back to GPU for face refinement
                self.refiner.to("cpu")
                self.pipeline.to("cuda")
                
            # Clear VRAM before face refinement
            torch.cuda.empty_cache()
            
            # Apply face refinement only if enabled
            if use_face_detailer:
                # Clear VRAM before face refinement
                torch.cuda.empty_cache()
                
                # Store original scale and set new scale for face refinement
                original_scale = scale  # Use the input scale parameter
                self.pipeline.set_ip_adapter_scale(0.7)  # Adjust this value as needed
                
                refined_images = []
                for image in images:
                    # Clear VRAM before each face refinement iteration
                    torch.cuda.empty_cache()
                    
                    refined_image = self.refine_faces(
                        image=image, 
                        prompt=prompt, 
                        face_image1=face_image1, 
                        face_image2=face_image2,
                        strength=face_refinement_strength
                    )
                    refined_images.append(refined_image)
                    
                    # Optional: force garbage collection after each iteration
                    torch.cuda.empty_cache()
                
                images = refined_images
                
                # Restore original IP-Adapter scale
                self.pipeline.set_ip_adapter_scale(original_scale)

            # Save each refined image
            for i, image in enumerate(images):
                output_path = f"./outputs/output-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

        return output_paths 