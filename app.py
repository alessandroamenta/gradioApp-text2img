import os
import io
import requests
import time
import json
import base64
from PIL import Image, ImageEnhance
from dotenv import load_dotenv, find_dotenv
import gradio as gr 

# Load the local .env file
load_dotenv(find_dotenv())

# Define the API key and URL
api_key = os.getenv('YOUR_API_KEY')
api_base = os.getenv('YOUR_API_BASE')


def get_completion(inputs, parameters=None, ENDPOINT_URL=api_base):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
        # Print the headers of the response
    print("Response Headers:", response.headers)

    # Print the first 100 characters of the response content
    print("Response Content:", response.content[:100])
    return response.content  # return the content as bytes

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(prompt, params)
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(output))

    return image  # return the PIL Image object


with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt")  # Give prompt some real estate
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit")  # Submit button side by side!
    with gr.Accordion("Advanced options", open=False):  # Let's hide the advanced options!
            negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
                    guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result")  # Move the output up too
            
    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

demo.queue(concurrency_count=80, max_size=100).launch(max_threads=150, share=True, server_port=int(os.getenv('PORT1')))
