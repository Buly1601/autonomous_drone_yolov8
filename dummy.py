import cv2
import numpy as np


def get_blue():
        
    # Load the image
    cv_image = cv2.imread("/home/buly/Desktop/drone_competition/runs/detect/predict5/image0.jpg")
    cv_image = cv2.resize(cv_image, (340, 340))  # Resize for consistent display

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for blue in HSV
    lower_blue = np.array([110, 150, 150])
    upper_blue = np.array([130, 255, 255])

    # Create a mask that keeps only the blue regions
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Create a black background of the same shape as the original image
    black_background = np.zeros_like(cv_image)

    # Use the mask to copy only the blue regions from the original image
    blue_only = cv2.bitwise_and(cv_image, cv_image, mask=mask)

    # Overlay the blue-only image onto the black background
    result = np.where(blue_only != 0, blue_only, black_background)
    return result


def get_square():
    result = get_blue()

    AREA = 0
    I = None

# Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(result, 30, 130)

    # Define a kernel (structuring element) for dilation
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel to control thickness

    # Dilate the edges to make them thicker
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
        # Display the result
    cv2.imshow("Blue Regions Only", thick_edges)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Find contours in the edges
    contours, _ = cv2.findContours(thick_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the same size as the result, initialized to black
    mask = np.zeros_like(result)

    # Iterate over contours to find and keep squares
    for i,contour in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 4 sides and is convex
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > AREA:
                AREA = area
                I = approx
            # Filter by area to avoid very small or large shapes
        
        # Fill the detected square on the mask
        cv2.drawContours(mask, [I], -1, (255, 255, 255), 150)

    if I is not None:
        # Get the center of the square (centroid)
        M = cv2.moments(I)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # X coordinate of the centroid
            cY = int(M["m01"] / M["m00"])  # Y coordinate of the centroid

            # Draw a red circle at the center of the square
            cv2.circle(result, (cX, cY), radius=5, color=(0, 0, 255), thickness=-1)  # Red color

    # Apply the mask to the original frame
    result = cv2.bitwise_and(result, mask)

    # Display the result
    cv2.imshow("Blue Regions Only", result)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows

get_square()
