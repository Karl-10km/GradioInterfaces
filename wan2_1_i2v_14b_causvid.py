# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from safetensors import safe_open
from huggingface_hub import hf_hub_download

import warnings
import gradio as gr
warnings.filterwarnings('ignore')

CKPT_DIR = "../Wan2.1/Wan2.1-I2V-14B-720P"
RESOLUTION_LIST = ['720*1280', '1280*720', '480*832', '832*480']

# Button Functions
def load_model() -> str:
    global pipe
    try:
        print("Loading I2V-14B 720P model...", end='', flush=True)
        # Load main model
        # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
        MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

        image_encoder = CLIPVisionModel.from_pretrained(MODEL_ID, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0) # 8.0 if using CausVid, else 5.0
        pipe.to("cuda")

        # Load CausVid LoRA
        LORA_REPO_ID = "Kijai/WanVideo_comfy"
        LORA_FILENAME = "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"
        causvid_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)

        lora_weights = None
        with safe_open(causvid_path, framework="pt") as f:
            lora_weights = {key: f.get_tensor(key).to(torch.bfloat16) for key in f.keys()}

        # Fix LoRA key names for diffusers
        prefix = "diffusion_model."
        transformer_lora_weights = {}
        for key in lora_weights.keys():
            if key.endswith("lora_down.weight") and key.startswith(prefix) and "blocks." in key:
                base_name = key[len(prefix) :].replace("lora_down.weight", "weight")
                b_key = key.replace("lora_down.weight", "lora_up.weight")
                if b_key in lora_weights:
                    transformer_lora_weights[key.replace("lora_down.weight", "lora_A.weight")] = lora_weights[key]
                    transformer_lora_weights[b_key.replace("lora_up.weight", "lora_B.weight")] = lora_weights[b_key]

        pipe.load_lora_weights(transformer_lora_weights, adapter_name="wan_lora")
        pipe.set_adapters(["wan_lora"], adapter_weights=[0.5]) # 0.4~0.6 if I2V, 0.2~0.4 if T2V
        pipe.fuse_lora()

        print("done", flush=True)
        return "Model Loaded"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: {str(e)}"


def i2v_generation(prompt, image, resolution, fps, inference_steps, video_length, guide_scale, n_prompt, seed):
    if image is None:
        return None, "Error: Please upload an image for I2V generation."

    if prompt.strip() == "":
        return None, "Error: Please provide a prompt."

    try:
        print(f"Generating video with prompt: {prompt}")
        print(f"Parameters: resolution={resolution}, steps={sd_steps}, guide_scale={guide_scale}, shift={shift_scale}, seed={seed}")
        
        max_area = resolution.split('*')[0] * resolution.split('*')[1]
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height), resample=Image.LANCZOS)
        video_length = 5  # seconds, can be adjusted
        # Generate video
        video = pipe(
            prompt,
            image,
            height=int(resolution.split('*')[0]),
            width=int(resolution.split('*')[1]),
            #shift=shift_scale,
            num_frames=int(video_length * fps)+1,
            num_inference_steps=inference_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=-1,  # Random seed
        ).frames[0]

        # Save video
        #formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        #formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        #save_file = f"i2v_14b_{size.replace('*','x')}_{formatted_prompt}_{formatted_time}.mp4"
        save_file = "result.mp4"
        
        export_to_video(video, save_file, fps=16)

        return save_file, "Video generated successfully!"

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        return None, error_msg


# Gradio Interface
def gradio_interface():
    with gr.Blocks(title="Wan2.1 I2V-14B") as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.1 (I2V-14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Text and Image to Video Generation with Wan I2V-14B Model
                    </div>
                    """)

        with gr.Row():
            with gr.Column():
                # Input image
                input_image = gr.Image(
                    type="pil",
                    label="Upload Input Image",
                    elem_id="image_upload",
                    height=600
                )

                # Prompt
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate from the image",
                    lines=3
                )

                n_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Describe the negative prompt you want to add"
                )

                # Generate button
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                # Generation parameters
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label="Video Resolution",
                            choices=RESOLUTION_LIST,
                            value="720*1280"
                        )
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=1.0,
                            step=1)
                    with gr.Row():
                        video_length = gr.Slider(
                            label="video length (seconds)",
                            minimum=1,
                            maximum=5,
                            value=5.0,
                            step=1)
                        inference_steps = gr.Slider(
                            label="Inference Step",
                            minimum=2,
                            maximum=100,
                            step=1,
                            value=8)
                    with gr.Row():
                        fps = gr.Slider(
                            label="frame per second",
                            minimum=8,
                            maximum=16,
                            value=16.0,
                            step=8)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)

            with gr.Column():
                # Output
                output_video = gr.Video(
                    label='Generated Video', 
                    interactive=False, 
                    height=600
                )
                
                generation_status = gr.Textbox(
                    label="Generation Status",
                    interactive=False
                )

        generate_btn.click(
            fn=i2v_generation,
            inputs=[
                prompt, image, resolution, fps, inference_steps, video_length, guide_scale, n_prompt, seed
            ],
            outputs=[output_video, generation_status]
        )

    return demo


if __name__ == '__main__':
    load_model()  # Load model at startup

    # Launch Gradio interface
    demo = gradio_interface()
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=True
    )