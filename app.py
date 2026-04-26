import gradio as gr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, GenerationConfig
from PIL import Image, ImageOps
from jiwer import cer, wer
import tempfile
import os
from config import FT_MODEL_NAME, get_device

# ── Load fine-tuned model from Hugging Face Hub ──────────────────────────────
print("Loading model...")

processor = TrOCRProcessor.from_pretrained(FT_MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(FT_MODEL_NAME, torch_dtype=torch.float32)

# Fix token IDs
model.config.decoder_start_token_id = 1
model.config.pad_token_id = 1
model.config.eos_token_id = 2
model.generation_config = GenerationConfig(
    decoder_start_token_id=1,
    eos_token_id=2,
    pad_token_id=1,
    max_new_tokens=64,
    forced_eos_token_id=None,
    forced_bos_token_id=None,
)
model.decoder.config.decoder_start_token_id = 1
model.decoder.config.eos_token_id = 2

device = torch.device(get_device())
model.to(device)
model.eval()
print(f"Model loaded on {device}")


# ── Core prediction function ──────────────────────────────────────────────────
def predict(image: Image.Image) -> str:
    image = image.convert("RGB")
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device).float()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ── Gradio handler ────────────────────────────────────────────────────────────
def run(image, ground_truth):
    if image is None:
        return "", "", "", None

    predicted = predict(image)

    # Build txt file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
    tmp.write(predicted)
    tmp.close()
    txt_path = tmp.name

    # If ground truth provided, compute metrics
    if ground_truth and ground_truth.strip():
        gt = ground_truth.strip()
        char_error = cer([gt], [predicted])
        word_error = wer([gt], [predicted])
        metrics_str = f"CER: {char_error:.2%}    WER: {word_error:.2%}"
    else:
        metrics_str = "Provide ground truth text above to see CER / WER."

    return predicted, metrics_str, txt_path


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Handwriting Recognition") as demo:

    gr.Markdown("""
    # Handwriting Recognition
    Upload a cropped line of handwritten text to transcribe it using a fine-tuned TrOCR model.
    Optionally provide the ground truth to see character and word error rates.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Handwritten Image")
            gt_input = gr.Textbox(label="Ground Truth (optional)", placeholder="Type the correct text here to get CER / WER...")
            submit_button = gr.Button("Transcribe", variant="primary")

        with gr.Column():
            prediction_output = gr.Textbox(label="Predicted Text")
            metrics_output = gr.Textbox(label="Metrics")
            download_output = gr.File(label="Download as .txt")

    submit_button.click(
        fn=run,
        inputs=[image_input, gt_input],
        outputs=[prediction_output, metrics_output, download_output]
    )

    gr.Markdown("""
    ---
    **Tips for best results:**
    - Use a cropped single line of handwriting, not a full page
    - Dark pen on white paper works best
    - Good lighting with no shadows
    """)

if __name__ == "__main__":
    demo.launch()