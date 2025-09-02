import os
import gradio as gr
from PIL import Image
from io import BytesIO, BufferedReader
from typing import Optional, List, Dict, Tuple, Any

from image_process import crop_image_ratio, crop_resize_image, byte2pil, pil2byte, base64encoder

from dotenv import load_dotenv
load_dotenv()
apikey_replicate = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = apikey_replicate
import replicate

def edit_image(image, models: str, prompt_select: str, prompt_input: str):
    global image_edit_prompts
    # crop and resize
    img = crop_resize_image(image, 16/9, (720, 1280))
    input = {}
    pil_output = ""

    if models == "qwen/qwen-image-edit":
        input = {
            "image": f"data:image/png;base64,{base64encoder(img)}",
            "prompt": prompt_input if len(prompt_input) > 0 else image_edit_prompts[prompt_select]
        }
    elif models == "google/nano-banana":
        input = {
            "image_input": [f"data:image/png;base64,{base64encoder(img)}"],
            "prompt": prompt_input if len(prompt_input) > 0 else image_edit_prompts[prompt_select]
        }

    output = replicate.run(models, input=input)
    if isinstance(output, list):
        print("list")
        output = output[0]
    else:
        print("not list")
    pil_output = byte2pil(output)
    
    return pil_output

def analyze_image(image, model, prompt):
    # 이미지 분석 로직을 여기에 구현
    tag = "분석된 태그들"
    caption = "생성된 캡션"
    analyzed_prompt = "분석 결과 프롬프트"
    return tag, caption, analyzed_prompt

def generate_video(prompt):
    # 비디오 생성 로직을 여기에 구현
    return None  # 임시로 None 반환

image_edit_prompts = {
    "style(anime)": "Convert the uploaded photo into Japanese anime style art. Keep characters, background, and clothing similar to the original.",
    "style(sketch)": "Convert to pencil sketch with natural graphite lines, cross-hatching, and visible paper texture while maintaining all background details and character features",
    "style(oil)": "Transform to oil painting with visible brushstrokes, thick paint texture, and rich color depth while maintaining all background details and character features",
    "style(water)": "Convert to watercolor painting with transparent colors, paper texture, and natural paint flow effects while maintaining all background details and character features"
}

# Gradio 인터페이스 생성
def gradio_interface():
    with gr.Blocks(title="Gen AI Multi-Function Interface") as demo:
        gr.Markdown("# Gen AI Multi-Function Interface")
        
        # 첫 번째 행: 이미지 편집 섹션
        with gr.Row():
            with gr.Column(scale=1):
                input_image_edit = gr.Image(label="Input: Image", 
                                            type="pil",
                                            height=480)
            
            with gr.Column(scale=1):
                select_models = gr.Dropdown(
                    choices=["qwen/qwen-image-edit", "google/nano-banana", "Flux-Kontext-Pro"], 
                    label="Select: Models"
                )
                select_prompt_edit = gr.Dropdown(
                    choices=list(image_edit_prompts.keys()),#["Prompt 1", "Prompt 2", "Prompt 3"], 
                    label="Select: Prompt(to edit)"
                )
                input_prompt_edit = gr.Textbox(
                    label="Input: Prompt(to edit)", 
                    placeholder="편집할 프롬프트를 입력하세요",
                    value=None,
                    lines=8
                )
                edit_button = gr.Button("Edit Image", variant="primary")
            
            with gr.Column(scale=1):
                display_edited_image = gr.Image(label="Display: Image(Edited)", 
                                                type="pil",
                                                height=480)
        
        # 두 번째 행: 선택 컨트롤들
        with gr.Row():
            select_image = gr.Dropdown(
                choices=["Image 1", "Image 2", "Image 3"], 
                label="Select: Image"
            )
            select_model = gr.Dropdown(
                choices=["Analysis Model 1", "Analysis Model 2"], 
                label="Select: Model"
            )
            select_prompt_analyze = gr.Dropdown(
                choices=["Analysis Prompt 1", "Analysis Prompt 2"], 
                label="Select: Prompt"
            )
        
        # 세 번째 행: 이미지 분석 및 결과 표시
        with gr.Row():
            with gr.Column(scale=1):
                input_image_analyze = gr.Image(label="Input: Image", height=300)
            
            with gr.Column(scale=1):
                input_prompt_analyze = gr.Textbox(
                    label="Input: Prompt(to analyze)", 
                    placeholder="분석할 프롬프트를 입력하세요",
                    lines=8
                )
                analyze_button = gr.Button("Analyze Image", variant="primary")
            
            with gr.Column(scale=1):
                display_tag = gr.Textbox(label="Display: Tag", lines=2)
                display_caption = gr.Textbox(label="Display: Caption", lines=3)
                display_prompt_result = gr.Textbox(label="Display: Prompt", lines=3)
        
        # 네 번째 행: 비디오 생성
        with gr.Row():
            with gr.Column(scale=1):
                select_prompt_bottom = gr.Dropdown(
                    choices=["Prompt A", "Prompt B", "Prompt C"], 
                    label="Select: Prompt"
                )
                # Generation parameters
                with gr.Accordion("Generation Parameters", open=True):
                    with gr.Row():
                        resolution = gr.Dropdown(
                            label="Video Resolution",
                            choices=["720*1280", "1280*720"],
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
                            minimum=4,
                            maximum=100,
                            step=1,
                            value=8)
                    with gr.Row():
                        fps = gr.Slider(
                            label="frame per second",
                            minimum=8,
                            maximum=16,
                            value=16,
                            step=4)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1)
            with gr.Column(scale=1):
                input_prompt_video = gr.Textbox(
                    label="Input: Prompt(to analyze)", 
                    placeholder="비디오 생성을 위한 프롬프트를 입력하세요",
                    lines=8
                )
                generate_button = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1):
                display_video = gr.Video(label="Display: Video", height=480)
        
        # 이벤트 핸들러 연결
        edit_button.click(
            fn=edit_image,
            inputs=[input_image_edit, select_models, select_prompt_edit, input_prompt_edit],
            outputs=display_edited_image
        )
        
        analyze_button.click(
            fn=analyze_image,
            inputs=[input_image_analyze, select_model, input_prompt_analyze],
            outputs=[display_tag, display_caption, display_prompt_result]
        )
        
        generate_button.click(
            fn=generate_video,
            inputs=input_prompt_video,
            outputs=display_video
        )

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch(
        #server_name='0.0.0.0',
        #server_port=7860,
        #share=True
    )