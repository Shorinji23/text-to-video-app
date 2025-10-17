from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import gradio as gr

# Load model (pertama kali jalan, bakal download ~5GB - sabar ya!)
print("Loading model... This may take a few minutes on first run.")
video_gen = pipeline(
    task=Tasks.text_to_video_synthesis,
    model='damo/text-to-video-synthesis'
)

def generate_video(prompt):
    print(f"Generating video for: {prompt}")
    result = video_gen(prompt)
    return result['output_video']

# Gradio interface
with gr.Blocks(title="My Video Generator") as demo:
    gr.Markdown("## 🎥 Text-to-Video Generator (ModelScope + HF Spaces)")
    gr.Markdown("Powered by `damo/text-to-video-synthesis`")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Prompt",
                value="A cat wearing sunglasses playing guitar on a beach"
            )
            run_btn = gr.Button("🎬 Generate Video", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Result", interactive=False)
    
    run_btn.click(fn=generate_video, inputs=text_input, outputs=video_output)
    gr.Markdown("💡 First run takes ~1-2 minutes. Subsequent runs are faster!")

demo.launch()
