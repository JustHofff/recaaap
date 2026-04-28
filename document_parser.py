# document_parser.py
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

MAX_WIDTH = 1500
PADDING = 8
MIN_BLOB_AREA = 20
MIN_LINE_WIDTH = 50
MIN_LINE_HEIGHT = 15
PDF_RENDER_SCALE = 2.0


def create_incremented_dir(base_path, prefix=""):
    counter = 1
    while True:
        dir_name = f"{prefix}{counter:03d}"
        full_path = os.path.join(base_path, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        counter += 1


def load_image(path):
    """
    Accepts a file path (str or Path) to a JPEG, PNG, TIFF, or PDF.
    Returns a list of BGR numpy arrays.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return [img]

    elif suffix == ".pdf":
        import fitz
        pages = []
        doc = fitz.open(str(path))
        for page in doc:
            mat = fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape(pix.h, pix.w, pix.n)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            pages.append(img_bgr)
        doc.close()
        return pages

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected jpg, png, tiff, or pdf.")


def preprocess(img_color, max_width=MAX_WIDTH):
    """
    Takes a BGR numpy array.
    Returns (img_color_resized, img_binary).
    """
    h, w = img_color.shape[:2]
    if w > max_width:
        scale = max_width / w
        img_color = cv2.resize(img_color, (max_width, int(h * scale)))

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(
        img_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return img_color, img_binary


def filter_lines(img_color, img_binary):
    """
    Removes ruled paper lines (blue/red) from img_binary using color detection.
    Returns a cleaned copy of img_binary.
    """
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0,   40, 40])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(img_hsv, lower_red1, upper_red1),
        cv2.inRange(img_hsv, lower_red2, upper_red2)
    )

    mask_lines  = cv2.bitwise_or(mask_blue, mask_red)
    img_cleaned = img_binary.copy()
    img_cleaned[mask_lines > 0] = 0
    return img_cleaned


def find_text_lines(img_cleaned, min_blob_area=MIN_BLOB_AREA):
    """
    Finds bounding boxes for each line of text using ink-row projection.
    Returns a list of (x, y, w, h) tuples sorted top to bottom.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        img_cleaned, connectivity=8
    )

    blobs = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_blob_area:
            blobs.append((x, y, w, h))

    if not blobs:
        return []

    img_h = img_cleaned.shape[0]
    ink_rows = np.zeros(img_h, dtype=np.int32)
    for (x, y, w, h) in blobs:
        ink_rows[y:y + h] += 1

    kernel_size = max(5, img_h // 60)
    ink_rows_smooth = np.convolve(ink_rows, np.ones(kernel_size), mode="same")

    in_line = False
    line_ranges = []
    start = 0
    for idx, val in enumerate(ink_rows_smooth):
        if not in_line and val > 0:
            in_line, start = True, idx
        elif in_line and val == 0:
            in_line = False
            line_ranges.append((start, idx))
    if in_line:
        line_ranges.append((start, img_h))

    line_boxes = []
    for (y0, y1) in line_ranges:
        in_range = [b for b in blobs if y0 <= b[1] + b[3] / 2 <= y1]
        if not in_range:
            continue
        x0 = min(b[0] for b in in_range)
        x1 = max(b[0] + b[2] for b in in_range)
        lw, lh = x1 - x0, y1 - y0
        if lw >= MIN_LINE_WIDTH and lh >= MIN_LINE_HEIGHT:
            line_boxes.append((x0, y0, lw, lh))

    return line_boxes


def crop_lines(img_color, line_boxes, output_dir, padding=PADDING):
    """
    Crops each line box from img_color and saves as a PNG to output_dir.
    Returns a list of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_color.shape[:2]
    saved_paths = []

    for i, (x, y, bw, bh) in enumerate(line_boxes):
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
        crop = img_color[y0:y1, x0:x1]
        path = output_dir / f"line_{i+1:03d}.png"
        cv2.imwrite(str(path), crop)
        saved_paths.append(path)

    return saved_paths


def parse_document(image_path, output_dir, max_width=MAX_WIDTH, padding=PADDING, min_blob_area=MIN_BLOB_AREA):
    """
    Full pipeline for file-based usage: load → preprocess → filter → find → crop.
    Saves cropped line images to output_dir.
    Returns a list of saved file paths.
    """
    pages = load_image(image_path)
    all_paths = []

    for page_num, page_img in enumerate(pages):
        img_color, img_binary = preprocess(page_img, max_width=max_width)
        print(f"[1/4] Loaded page from {image_path}")

        img_cleaned = filter_lines(img_color, img_binary)
        print(f"[2/4] Preprocessed — image size: {img_color.shape[:2]}")

        line_boxes = find_text_lines(img_cleaned, min_blob_area=min_blob_area)
        print(f"[3/4] Found {len(line_boxes)} text lines")

        page_dir = create_incremented_dir(Path(output_dir), "page_")
        paths = crop_lines(img_color, line_boxes, page_dir, padding=padding)
        print(f"[4/4] Saved {len(paths)} crops to {page_dir}")

        all_paths.extend(paths)

    return all_paths


def parse_pil_image(pil_image, padding=PADDING, min_blob_area=MIN_BLOB_AREA):
    """
    In-memory pipeline for app usage: PIL image → list of PIL line crops.
    No files written to disk.
    """
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_color, img_bin = preprocess(img_bgr)
    img_cleaned = filter_lines(img_color, img_bin)
    line_boxes = find_text_lines(img_cleaned, min_blob_area=min_blob_area)

    h, w = img_color.shape[:2]
    crops = []
    for (x, y, bw, bh) in line_boxes:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
        crop_rgb = cv2.cvtColor(img_color[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop_rgb))

    return crops