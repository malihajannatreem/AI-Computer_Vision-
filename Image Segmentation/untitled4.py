import cv2
import numpy as np
import os


def segment_person(image_path):
    # Step 1: Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file at {image_path} does not exist.")
        return

    # Step 2: Load the Image
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load the image at {image_path}. Please check the file format and path.")
        return

    # Step 3: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 5: Use Otsu's thresholding to create a binary mask
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Create a mask for the detected person
    person_mask = np.zeros_like(binary_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(person_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Step 8: Create the final binary image
    final_mask = np.where(person_mask > 0, 255, 0).astype(np.uint8)

    # Step 9: Save or Display the Result
    cv2.imwrite('segmented_person.png', final_mask)
    
    print("Segmentation completed and saved as 'segmented_person.png'.")

# Example usage
image_path = r"C:\Users\Nisal Dias\Documents\IMG_87631.JPG"
segment_person(image_path)