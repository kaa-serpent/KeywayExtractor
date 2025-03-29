import cv2
import numpy as np
import os
import sys

# Path to the folder containing keyway images
KEYWAY_FOLDER = "keywayCleaned"

# Get the image path from command line arguments
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Exaggerate brightness by applying a linear transformation (contrast and brightness)
alpha = 3.0  # Increase contrast (1.0 means no change)
beta = 80     # Increase brightness
brightened_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Apply Gaussian Blur to reduce noise

blurred = cv2.GaussianBlur(brightened_image, (5, 5), 0)

# Apply thresholding to isolate the keyway area
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size to find the keyway region
min_area = 300  # Adjust the minimum area as needed
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

if filtered_contours:
    # Find the largest contour (assuming it's the keyway)
    contour = max(filtered_contours, key=cv2.contourArea)

    # Get bounding box around the keyway
    x, y, w, h = cv2.boundingRect(contour)

    # Crop the image around the keyway
    cropped_image = image[y:y+h, x:x+w]

    # Convert cropped image to grayscale
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_cropped = clahe.apply(gray_cropped)

    # Thresholding to isolate the darker keyway areas (including pin)
    lower_thresh = 0   # Adjust based on the darkness of the keyway
    upper_thresh = 60  # Adjust to include pin and exclude rotor brightness

    # Apply the threshold
    thresh_cropped = cv2.inRange(enhanced_cropped, lower_thresh, upper_thresh)

    # Apply morphological operations to clean up the image
    kernel = np.ones((5, 5), np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh_cropped, cv2.MORPH_CLOSE, kernel)

    # Invert the thresholded image
    inverted_thresh = cv2.bitwise_not(cleaned_thresh)
    cv2.imwrite("inverted_keyway.jpg", inverted_thresh)

    # Load the extracted keyway shape for comparison
    extracted_keyway = cv2.imread("inverted_keyway.jpg", cv2.IMREAD_GRAYSCALE)
    if extracted_keyway is None:
        raise ValueError("Failed to load the extracted keyway image.")

    # Apply Otsu's Thresholding again for consistency
    _, extracted_keyway = cv2.threshold(extracted_keyway, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Rotate if taller than wide
    h, w = extracted_keyway.shape
    if h > w:
        extracted_keyway = cv2.rotate(extracted_keyway, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Normalize the extracted keyway image
    extracted_keyway = extracted_keyway.astype(np.float32) / 255.0

    # Compare with keyway images in the folder
    similarities = []
    for filename in os.listdir(KEYWAY_FOLDER):
        file_path = os.path.join(KEYWAY_FOLDER, filename)
        keyway_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if keyway_img is None:
            continue

        # Apply Otsu's Thresholding
        _, keyway_img = cv2.threshold(keyway_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Rotate if taller than wide
        h, w = keyway_img.shape
        if h > w:
            keyway_img = cv2.rotate(keyway_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Normalize the keyway image
        keyway_img = keyway_img.astype(np.float32) / 255.0

        # Auto-scaling: Resize keyway image to match the size of the extracted keyway
        keyway_img_resized = cv2.resize(keyway_img, (extracted_keyway.shape[1], extracted_keyway.shape[0]), interpolation=cv2.INTER_AREA)

        # Apply Normalized Cross-Correlation (NCC)
        result = cv2.matchTemplate(keyway_img_resized, extracted_keyway, method=cv2.TM_CCOEFF_NORMED)

        # Get the maximum value from the result, which is the similarity score
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        similarity = max_val  # Higher value means better match

        similarities.append((filename, similarity))

        # Stop early if similarity is 100%
        if similarity == 1.0:
            break

    # Sort by similarity and keep the top 5
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_matches = similarities[:5]

    for match in top_5_matches:
        print(f"{os.path.join(KEYWAY_FOLDER, match[0])} - Similarity Score: {match[1]:.6f}")
        # print(f"{match[0]} - Similarity Score: {match[1]:.6f}")
else:
    print("No contours detected. Try adjusting the threshold or morphological operations.")
