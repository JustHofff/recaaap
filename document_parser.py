# document_parser.py
import cv2
import numpy as np
from pathlib import Path

MAX_WIDTH = 1500
PADDING = 8
MIN_BLOB_AREA = 20
MIN_LINE_WIDTH = 50
MIN_LINE_HEIGHT = 15
MERGE_THRESHOLD_FACTOR = 0.8
PDF_RENDER_SCALE = 2.0

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
            # PyMuPDF gives us RGB — convert to BGR for OpenCV
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
    Takes a BGR numpy array from load_image().
    Returns (img_color_resized, img_binary)
    """
    h, w = img_color.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        img_color = cv2.resize(img_color, (new_w, new_h))

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    _, img_binary = cv2.threshold(
        img_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return img_color, img_binary

def filter_lines(img_color, img_binary):
    """
    Removes paper lines from img_binary using color detection.
    Returns a cleaned copy of img_binary with ruled line pixels set to 0.
    """
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

    # Blue lines (notebook paper)
    lower_blue = np.array([100, 40, 40])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # Red lines (margin lines)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # Combine all masks
    mask_lines = cv2.bitwise_or(mask_blue, mask_red1)
    mask_lines = cv2.bitwise_or(mask_lines, mask_red2)

    # Remove masked pixels from binary image
    img_cleaned = img_binary.copy()
    img_cleaned[mask_lines > 0] = 0

    return img_cleaned

def find_text_lines(img_cleaned, min_blob_area=MIN_BLOB_AREA):
    """
    Finds bounding boxes for each line of text in img_cleaned.
    min_blob_area filters out specks and noise.
    Returns a list of (x, y, w, h) tuples, sorted top to bottom.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        img_cleaned, connectivity=8
    )

    # Collect valid blobs, skipping label 0 (background)
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

    # Sort blobs top to bottom by their vertical center
    blobs.sort(key=lambda b: b[1] + b[3] / 2)

    # Compute merge threshold from median blob height
    heights = [b[3] for b in blobs]
    median_h = sorted(heights)[len(heights) // 2]
    merge_threshold = median_h * MERGE_THRESHOLD_FACTOR

    # Group blobs into lines
    lines = []
    current_group = [blobs[0]]

    # Compares against the group's full vertical span
    for blob in blobs[1:]:
        group_y0 = min(b[1] for b in current_group)
        group_y1 = max(b[1] + b[3] for b in current_group)
        group_center_y = (group_y0 + group_y1) / 2
        this_center_y = blob[1] + blob[3] / 2

        if abs(this_center_y - group_center_y) <= merge_threshold:
            current_group.append(blob)
        else:
            lines.append(current_group)
            current_group = [blob]
    lines.append(current_group)

    # Merge each group into one bounding box
    line_boxes = []
    for group in lines:
        x0 = min(b[0] for b in group)
        y0 = min(b[1] for b in group)
        x1 = max(b[0] + b[2] for b in group)
        y1 = max(b[1] + b[3] for b in group)
        line_boxes.append((x0, y0, x1 - x0, y1 - y0))

    # Filter out boxes that are too small to be real lines
    min_line_width = MIN_LINE_WIDTH
    min_line_height = MIN_LINE_HEIGHT

    line_boxes = [
        box for box in line_boxes
        if box[2] >= min_line_width and box[3] >= min_line_height
    ]

    return line_boxes

def crop_lines(img_color, line_boxes, output_dir, padding=PADDING):
    """
    Crops each line box from img_color and saves as a PNG.
    Returns a list of output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = img_color.shape[:2]
    saved_paths = []

    for i, (x, y, bw, bh) in enumerate(line_boxes):
        # Add padding, clamped to image boundaries
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
    Full pipeline: load → preprocess → filter lines → find text lines → crop.
    Accepts jpg, png, tiff, or pdf.
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

        # Give each page its own subdirectory for multi-page PDFs
        page_dir = Path(output_dir) / f"page_{page_num+1:03d}"
        paths = crop_lines(img_color, line_boxes, page_dir, padding=padding)
        print(f"[4/4] Saved {len(paths)} crops to {page_dir}")

        all_paths.extend(paths)

    return all_paths