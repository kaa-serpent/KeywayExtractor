import cv2
import os
import re
import shutil
from tqdm import tqdm
import pytesseract
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim

# Path to the PDF
pdf_path = r"C:\Users\Sico\Documents\projet\blankFinder\sources\silcaBlanks.pdf"

# Create and clean the "keyway" directory
keyway_dir = "keyway"
if os.path.exists(keyway_dir):
    shutil.rmtree(keyway_dir)
os.makedirs(keyway_dir)

# Load excluded shapes
exclude_dir = r"C:\Users\Sico\Documents\projet\blankFinder\sources\shapes_to_exclude"
excluded_shapes = []

for filename in os.listdir(exclude_dir):
    shape_path = os.path.join(exclude_dir, filename)
    img = cv2.imread(shape_path)
    if img is not None:
        excluded_shapes.append(img)

# Define batch processing
batch_size = 10  # Process 10 pages at a time
total_pages = 328  # Adjust according to your PDF

similarity_threshold = 0.70


def is_similar(image1, image2, threshold=similarity_threshold):
    """Compare deux images avec SSIM et retourne True si elles sont suffisamment similaires."""
    if image1.shape != image2.shape:
        return False
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score >= threshold


# Process the PDF in batches
for start_page in tqdm(range(1, total_pages + 1, batch_size),desc="Total "):
    end_page = min(start_page + batch_size - 1, total_pages)
    print(f"\n[INFO] Processing pages {start_page} to {end_page}...")

    pages = convert_from_path(pdf_path, dpi=300, first_page=start_page, last_page=end_page)

    for idx, page in enumerate(tqdm(pages, desc="Extracting Keyways ")):
        page_num = start_page + idx  # Calculate actual page number
        page_path = f"page_{page_num}.jpg"
        page.save(page_path, "JPEG")

        # Load image
        img = cv2.imread(page_path)
        height, width, _ = img.shape

        # Crop image (removing top/bottom margins)
        img = img[int(height * 0.16):int(height * 0.95), :]

        # Handle odd/even page layout cropping
        if page_num % 2 != 0:  # Odd pages → Keep right part
            img_cropped = img[:, int(width * 0.30):]
        else:  # Even pages → Keep left part
            img_cropped = img[:, :int(width * 0.70)]

        # Convert to grayscale and apply binary threshold
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find all contours (potential keyways)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) > 4:  # Likely a keyway (not simple noise)
                x, y, w, h = cv2.boundingRect(contour)

                if cv2.contourArea(contour) > 600 and w > h and (float(w) / h) < 6:
                    dark_shape_crop = img_cropped[y:y + h, x:x + w]

                    # Define text extraction region below keyway
                    roi_y_start, roi_y_end = y + h, y + h + 60
                    roi_x_start, roi_x_end = max(x - 20, 0), min(x + w + 20, img_cropped.shape[1])
                    text_roi = img_cropped[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                    # OCR for keyway identification
                    extracted_text = pytesseract.image_to_string(text_roi, config="--psm 6").strip().replace("\n", " ")
                    extracted_text = re.sub(r"[\\/:*?\"<>|]", "", extracted_text)  # Remove invalid filename chars

                    # Generate filename
                    dark_shape_filename = (
                        os.path.join(keyway_dir, f"{extracted_text}.jpg") if extracted_text else
                        os.path.join(keyway_dir, f"keyway_{page_num}_{i}.jpg")
                    )

                    # Check if shape matches an excluded template
                    is_excluded = any(is_similar(dark_shape_crop, excl) for excl in excluded_shapes)

                    if is_excluded:
                        pass
                        # print(f"Skipped excluded shape {i}. (Similarity > {similarity_threshold * 100:.0f}%)")

                    else:
                        cv2.imwrite(dark_shape_filename, dark_shape_crop)

    print(f"\n[INFO] Finished processing batch: {start_page} to {end_page}")
