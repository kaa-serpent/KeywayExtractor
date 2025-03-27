import cv2
import os
import re
import shutil
import pytesseract
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim


# Path to the PDF
pdf_path = r"C:\Users\Sico\Documents\projet\blankFinder\sources\silcaBlanks.pdf"

# Create and clean the "keyway" directory
keyway_dir = "keyway"
if os.path.exists(keyway_dir):
    # Clean the folder by removing all files
    shutil.rmtree(keyway_dir)
    os.makedirs(keyway_dir)  # Recreate the folder
else:
    os.makedirs(keyway_dir)  # Create the folder if it doesn't exist

# Convert specific page (you can adjust first_page and last_page if necessary)
pages = convert_from_path(pdf_path, dpi=300)  # , first_page=19, last_page=30)  # First page as test


def is_similar(image1, image2, threshold=0.8):
    """Compare two images using Structural Similarity Index (SSIM) and return True if they are similar."""
    if image1.shape != image2.shape:
        return False  # Avoid comparing images of different sizes

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(gray1, gray2, full=True)
    return score > threshold


# Load excluded shapes
exclude_dir = r"C:\Users\Sico\Documents\projet\blankFinder\sources\shapes_to_exclude"
excluded_shapes = []

for filename in os.listdir(exclude_dir):
    shape_path = os.path.join(exclude_dir, filename)
    img = cv2.imread(shape_path)
    if img is not None:
        excluded_shapes.append(img)

# Loop through pages (for testing purposes, let's handle one page first)
for page_num, page in enumerate(pages, start=1):
    # Save page as image
    first_page_path = f"page_{page_num}.jpg"
    page.save(first_page_path, "JPEG")

    # Load image
    img = cv2.imread(first_page_path)

    # Get image dimensions
    height, width, _ = img.shape

    # Calculate the crop region (16% from top and bottom)
    top_crop = int(height * 0.16)
    bottom_crop = int(height * 0.95)

    # Crop the image horizontally (top and bottom)
    img = img[top_crop:bottom_crop, :]

    # For odd pages (1, 3, 5, etc.), crop 30% from left and keep the right
    if page_num % 2 != 0:  # Odd pages
        crop_x_start = int(width * 0.30)  # 30% from the left, keep right
        img_cropped = img[:, crop_x_start:]  # Keep right part
    else:  # Even pages
        crop_x_end = int(width * 0.70)  # 30% from the right, keep left
        img_cropped = img[:, :crop_x_end]  # Keep left part

    # Convert to grayscale
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding to isolate dark shapes (keyways, blanks, etc.)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # Invert binary image for dark shapes

    # Find all contours (dark shapes)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and extract each dark shape
    for i, contour in enumerate(contours):
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)  # You can adjust epsilon for approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter out contours with long straight lines (likely key blanks)
        # We will ignore contours that have more than a few points or are nearly straight
        if len(approx) > 4:  # If the shape has more than 4 points, it's likely a non-linear shape (keyway)

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter based on the area of the contour (excluding small contours like text)
            contour_area = cv2.contourArea(contour)
            if contour_area > 600:  # Exclude small contours (like text)
                aspect_ratio = float(w) / h if h != 0 else 0  # Aspect ratio of bounding box

                # Exclude elongated shapes (likely text)
                if aspect_ratio < 6:  # Adjust this threshold as needed
                    # Remove shapes where height is larger than width (likely noise)
                    if w > h:  # Keyways are wider than tall
                        dark_shape_crop = img_cropped[y:y + h, x:x + w]

                        # OCR to extract text from the area just below the keyway
                        roi_y_start = y + h
                        roi_y_end = roi_y_start + 60  # Define a region below the keyway to extract text
                        roi_x_start = x - 20
                        roi_x_end = x + w + 20  # Define a region below the keyway to extract text

                        # Crop the region where text should be located
                        text_roi = img_cropped[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                        # Use OCR to extract text from this region
                        extracted_text = pytesseract.image_to_string(text_roi, config="--psm 6")

                        # Clean up the text (remove unwanted characters, spaces, etc.)
                        extracted_text = extracted_text.strip().replace("\n", " ")

                        # If there’s any text, use it as the keyway name
                        if extracted_text:
                            extracted_text = re.sub(r"[\\/:*?\"<>|]", "", extracted_text)
                            dark_shape_filename = os.path.join(keyway_dir, f"{extracted_text}.jpg")
                        else:
                            dark_shape_filename = os.path.join(keyway_dir, f"keyway_{page_num}_{i}.jpg")

                        # Save the cropped keyway with the extracted name
                        # Check if the extracted keyway matches any excluded shape
                        is_excluded = any(is_similar(dark_shape_crop, excl) for excl in excluded_shapes)

                        if not is_excluded:
                            cv2.imwrite(dark_shape_filename, dark_shape_crop)
                            print(f"[INFO] Dark shape {i} saved as {dark_shape_filename}")
                        else:
                            print(f"[INFO] Skipped excluded shape {i}.")
                    else:
                        print(f"[INFO] Ignored contour {i} because it is taller than wide.")
                else:
                    print(f"[INFO] Ignored contour {i} due to aspect ratio or size.")
            else:
                print(f"[INFO] Ignored contour {i} due to small area.")

    print(f"[INFO] Processed page {page_num}")
