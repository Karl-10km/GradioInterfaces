# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import os.path as osp
import sys
import warnings
from datetime import datetime

import gradio as gr

warnings.filterwarnings('ignore')

# Model
sys.path.insert(
    0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'Wan2.1'))
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video

CKPT_DIR = "../Wan2.1/Wan2.1-I2V-14B-720P"

# Global Variables
wan_i2v_14b = None

# Button Functions
def load_model():
    global wan_i2v_14b
    try:
        print("Loading I2V-14B 720P model...", end='', flush=True)
        cfg = WAN_CONFIGS['i2v-14B']
        wan_i2v_14b = wan.WanI2V(
            config=cfg,
            checkpoint_dir=CKPT_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False
            #t5_cpu=False,
            #convert_model_dtype=False,
        )
        print("done", flush=True)
        return "Model Loaded"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"Error: {str(e)}"


def i2v_generation(prompt, image, resolution, sd_steps, guide_scale, shift_scale, seed, n_prompt):
    if wan_i2v_14b is None:
        return None, "Error: Model not loaded. Please load the model first."

    if image is None:
        return None, "Error: Please upload an image for I2V generation."

    if prompt.strip() == "":
        return None, "Error: Please provide a prompt."

    try:
        print(f"Generating video with prompt: {prompt}")
        print(f"Parameters: resolution={resolution}, steps={sd_steps}, guide_scale={guide_scale}, shift={shift_scale}, seed={seed}")
        
        # Generate video
        video = wan_i2v_14b.generate(
            prompt,
            image,
            max_area=MAX_AREA_CONFIGS[resolution],
            shift=shift_scale,
            sampling_steps=sd_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=True
        )

        # Save video
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = f"wan21_i2v_14b_{resolution.replace('*','x')}_{formatted_time}.mp4"
        
        cfg = WAN_CONFIGS['i2v-14B']
        cache_video(
            tensor=video[None],
            save_file=save_file,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

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

                resolution = gr.Dropdown(
                    label="Video Resolution",
                    choices=list(SUPPORTED_SIZES['i2v-14B']),
                    value="720*1280"
                )

                # Generate button
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                # Generation parameters
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        sd_steps = gr.Slider(
                            label="Diffusion steps",
                            minimum=1,
                            maximum=1000,
                            value=50,
                            step=1)
                        guide_scale = gr.Slider(
                            label="Guide scale",
                            minimum=0,
                            maximum=20,
                            value=5.0,
                            step=1)
                    with gr.Row():
                        shift_scale = gr.Slider(
                            label="Shift scale",
                            minimum=0,
                            maximum=10,
                            value=5.0,
                            step=1)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
                    n_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Describe the negative prompt you want to add"
                    )

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
                prompt_input, input_image, resolution, sd_steps,
                guide_scale, shift_scale, seed, n_prompt
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