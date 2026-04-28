import gradio as gr
import torch
import tempfile
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps
from config import PT_MODEL_NAME
from document_parser import parse_pil_image

# Load model
print("Loading model...")
processor = TrOCRProcessor.from_pretrained(PT_MODEL_NAME)
model     = VisionEncoderDecoderModel.from_pretrained(PT_MODEL_NAME)
model.eval()
print("Model loaded.")


# OCR

def predict_line(pil_image):
    image = pil_image.convert("RGB")
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.float()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# Gradio handler

def transcribe(pil_image):
    if pil_image is None:
        return [], "", None

    line_crops = parse_pil_image(pil_image)
    if not line_crops:
        return [], "No text lines detected. Try a cleaner scan.", None

    results = []
    full_lines = []

    for i, crop in enumerate(line_crops):
        text = predict_line(crop)
        full_lines.append(text)
        results.append((crop, f"Line {i+1}: {text}"))

    full_text = "\n".join(full_lines)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    tmp.write(full_text)
    tmp.close()

    return results, full_text, tmp.name


# UI

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap');

:root {
    --ink:        #1a1a2e;
    --paper:      #f5f0e8;
    --cream:      #ede8dc;
    --accent:     #c0392b;
    --rule:       #d4ccbb;
    --mono:       'DM Mono', monospace;
    --serif:      'DM Serif Display', serif;
}

body, .gradio-container {
    background-color: var(--paper) !important;
    font-family: var(--mono) !important;
    color: var(--ink) !important;
}

#header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 2rem;
}

#header h1 {
    font-family: var(--serif) !important;
    font-size: 2.8rem !important;
    color: var(--ink) !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
}

#header p {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    color: #666 !important;
    margin-top: 0.4rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.upload-zone {
    border: 2px dashed var(--rule) !important;
    border-radius: 4px !important;
    background: var(--cream) !important;
    min-height: 320px !important;
}

.upload-zone:hover {
    border-color: var(--ink) !important;
}

.upload-zone span.svelte-1vmd51o,
.upload-zone .icon-wrap.svelte-1vmd51o svg {
    color: var(--ink) !important;
    stroke: var(--ink) !important;
}

.upload-zone * {
    color: var(--ink) !important;
}

#transcribe-btn {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--mono) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    margin-top: 0.75rem !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
}

#transcribe-btn:hover {
    background: var(--accent) !important;
}

#lines-gallery .gallery-item {
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    background: white !important;
}

#full-text textarea {
    font-family: var(--mono) !important;
    font-size: 1rem !important;
    line-height: 1.8 !important;
    background: white !important;
    border: 1px solid var(--rule) !important;
    border-radius: 3px !important;
    color: var(--ink) !important;
    min-height: 200px !important;
    padding: 1rem !important;
}

label {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 0.4rem !important;
}

.divider {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 1.5rem 0;
}

#footer {
    text-align: center;
    font-size: 0.75rem;
    color: #aaa;
    padding: 1.5rem 0;
    border-top: 1px solid var(--rule);
    margin-top: 2rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
"""

with gr.Blocks(title="Handwriting Recognition") as demo:

    with gr.Column(elem_id="header"):
        gr.HTML("<h1>Handwriting Recognition</h1>")
        gr.HTML("<p>Upload a scanned handwritten document — get back clean text</p>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                elem_classes=["upload-zone"],
            )
            transcribe_btn = gr.Button(
                "Transcribe Document",
                variant="primary",
                elem_id="transcribe-btn",
            )

        with gr.Column(scale=2):
            lines_gallery = gr.Gallery(
                label="Detected Lines",
                columns=1,
                object_fit="contain",
                height=340,
                elem_id="lines-gallery",
            )
            gr.HTML("<hr class='divider'>")
            full_text_output = gr.Textbox(
                label="Full Transcription",
                lines=8,
                elem_id="full-text",
            )
            download_output = gr.File(label="Download as .txt")

    transcribe_btn.click(
        fn=transcribe,
        inputs=[image_input],
        outputs=[lines_gallery, full_text_output, download_output],
    )

    gr.HTML("""
        <div id='footer'>
            Powered by Microsoft TrOCR &nbsp;·&nbsp; Built with Gradio
        </div>
    """)

if __name__ == "__main__":
    demo.launch(css=CSS)