# RECAAAP
### Handwritten Text Recognition with TrOCR

Fine-tuning Microsoft's TrOCR model for handwriting recognition, with a Gradio app for transcribing handwritten documents.

---

## Results

| Model | CER | WER |
|---|---|---|
| Pretrained TrOCR (IAM baseline) | 2.54% | 6.74% |
| Pretrained TrOCR (Kaggle dataset) | 13.69% | 48.63% |
| Fine-tuned TrOCR (Kaggle dataset) | **6.77%** | **17.49%** |

---

## App

Upload a scanned handwritten document and get back a transcription line by line.

```bash
git clone git@github.com:JustHofff/recaaap.git
```
```bash
pip install -r requirements.txt
```
```bash
python app.py
```

---

## Model

Fine-tuned model available on Hugging Face: [`jhofff/trocr-finetuned-handwriting`](https://huggingface.co/jhofff/trocr-finetuned-handwriting)

---

## Stack

- [TrOCR](https://huggingface.co/microsoft/trocr-small-handwritten) — Microsoft's transformer-based OCR model
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://gradio.app)
- [OpenCV](https://opencv.org) — document line segmentation