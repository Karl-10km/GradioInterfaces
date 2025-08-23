# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import random

import torch
import gradio as gr
from PIL import Image

warnings.filterwarnings('ignore')

# Model
sys.path.insert(
    0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'Wan2.2'))
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video

def crop_image_ratio(img, target_aspect_ratio):
    # Get original dimensions
    original_width, original_height = img.size

    # Calculate new dimensions for cropping
    if original_height / original_width < target_aspect_ratio:
        # Original image is wider than target, crop width
        new_width = int(original_height / target_aspect_ratio)
        new_height = original_height
    else:
        # Original image is taller than target, crop height
        new_height = int(original_width * target_aspect_ratio)
        new_width = original_width

    # Calculate crop box coordinates for center crop
    left = (original_width - new_width) / 2
    upper = (original_height - new_height) / 2
    right = left + new_width
    lower = upper + new_height

    # Ensure coordinates are integers
    left, upper, right, lower = int(left), int(upper), int(right), int(lower)

    # Perform the crop
    cropped_image = img.crop((left, upper, right, lower))
    return cropped_image

# Button Functions
def load_model():
    """
    Args:
        config (EasyDict):
            Object containing model parameters initialized from config.py
        checkpoint_dir (`str`):
            Path to directory containing model checkpoints
        device_id (`int`,  *optional*, defaults to 0):
            Id of target GPU device
        rank (`int`,  *optional*, defaults to 0):
            Process rank for distributed training
        t5_fsdp (`bool`, *optional*, defaults to False):
            Enable FSDP sharding for T5 model
        dit_fsdp (`bool`, *optional*, defaults to False):
            Enable FSDP sharding for DiT model
        use_sp (`bool`, *optional*, defaults to False):
            Enable distribution strategy of sequence parallel.
        t5_cpu (`bool`, *optional*, defaults to False):
            Whether to place T5 model on CPU. Only works without t5_fsdp.
        init_on_cpu (`bool`, *optional*, defaults to True):
            Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        convert_model_dtype (`bool`, *optional*, defaults to False):
            Convert DiT model parameters dtype to 'config.param_dtype'.
            Only works without FSDP.
    """
    global wan_i2v_a14b
    try:
        print("Loading I2V-A14B model...", end='', flush=True)
        cfg = WAN_CONFIGS['i2v-A14B']
        wan_i2v_a14b = wan.WanI2V(
            config=cfg,
            checkpoint_dir='../Wan2.2/Wan2.2-I2V-A14B',
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=False,
        )
        print("done", flush=True)
        return "Model Loaded"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: {str(e)}"


def i2v_generation(prompt, image, resolution, fps, frame_num, sample_steps, 
                   guide_scale, shift_scale, seed, n_prompt, sample_solver):
    if wan_i2v_a14b is None:
        return None, "Error: Model not loaded. Please load the model first."

    if image is None:
        return None, "Error: Please upload an image for I2V generation."

    if prompt.strip() == "":
        return None, "Error: Please provide a prompt."

    try:
        print(f"Generating video with prompt: {prompt}")
        print(f"Parameters: resolution={resolution}, fps={fps}, frame_num={frame_num+1}, steps={sample_steps}, guide_scale={guide_scale}, shift={shift_scale}, seed={seed}")
        
        # crop to target aspect ratio then resize
        target_width, target_height = map(int, resolution.split('*'))
        target_ratio = target_height / target_width
        cropped_image = crop_image_ratio(image, target_ratio)
        resized_image = cropped_image.resize((target_width, target_height))
        print(f"width: {resized_image.width}, height: {resized_image.height}")

        # Generate video
        video = wan_i2v_a14b.generate(
            prompt,
            resized_image,
            max_area=MAX_AREA_CONFIGS[resolution],
            frame_num=frame_num+1,
            shift=shift_scale,
            #sample_solver=sample_solver,
            #sampling_steps=sample_steps,
            #guide_scale=guide_scale,
            #n_prompt=n_prompt,
            #seed=seed,
            offload_model=True
        )

        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"wan22_i2v_a14b__{formatted_time}.mp4"
        #save_file = "result.mp4"
        
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

        return save_file, resized_image, "Video generated successfully!"

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        print(error_msg)
        return None, None, error_msg


# Gradio Interface
def gradio_interface():
    with gr.Blocks(title="Wan2.2 I2V-A14B") as demo:
        gr.Markdown("""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        Wan2.2 (I2V-A14B)
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        Image to Video Generation with Wan 2.2 I2V-A14B Model
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

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Describe what you don't want in the video",
                    lines=2
                )

                # Generate button
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                # Generation parameters
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label="Video Size",
                            choices=list(SUPPORTED_SIZES['i2v-A14B']),
                            value="720*1280"
                        )
                    with gr.Row():
                        fps = gr.Slider(
                            label="frame per second",
                            minimum=8,
                            maximum=32,
                            value=16,
                            step=2)
                        frame_num = gr.Slider(
                            label="Frame Number",
                            minimum=4,
                            maximum=120,
                            value=80,
                            step=4,
                            info="24 FPS is recommended for smooth video generation. Adjust based on your needs."
                        )                    
                    with gr.Row():
                        sample_steps = gr.Slider(
                            label="Sampling Steps",
                            minimum=1,
                            maximum=100,
                            value=40,
                            step=1
                        )
                        guide_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0.0,
                            maximum=20.0,
                            value=5.0,
                            step=0.5
                        )                 
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift Scale",
                            minimum=0.0,
                            maximum=10.0,
                            value=5.0,
                            step=0.1
                        )
                        seed = gr.Slider(
                            label="Seed (-1 for random)",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1
                        )                    
                    sample_solver = gr.Dropdown(
                        label="Sample Solver",
                        choices=["unipc", "dpm++"],
                        value="unipc"
                    )

            with gr.Column():
                # Output
                output_video = gr.Video(
                    label='Generated Video', 
                    interactive=False, 
                    height=600
                )

                resized_image = gr.Image(
                    label='Resized Image',
                    interactive=False,
                    type="pil",
                    elem_id="resized_image",
                    height=600)
                                
                generation_status = gr.Textbox(
                    label="Generation Status",
                    interactive=False
                )

        generate_btn.click(
            fn=i2v_generation,
            inputs=[
                prompt_input, input_image, resolution, fps, frame_num, sample_steps,
                guide_scale, shift_scale, seed, negative_prompt, sample_solver
            ],
            outputs=[output_video, resized_image, generation_status]
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