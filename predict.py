from ultralytics import YOLO
import cv2
import numpy as np

def predict():
    # Load the trained model
    model = YOLO("/home/buly/Desktop/drone_competition/best.pt")
    image=cv2.imread("/home/buly/Downloads/66.jpg")

    # Perform prediction on a custom image
    results = model.predict(source=image, save=True)


def get_biggest_square():
        """
        Function to get the biggest square from the frame
        """
        # Create a named window for displaying the video
        cv2.namedWindow("Bebop Camera", cv2.WINDOW_NORMAL)
        cv_image = cv2.imread("/home/buly/Desktop/drone_competition/runs/detect/predict/image0.jpg")
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the same size as the cv_image, initialized to black
        mask = np.zeros_like(cv_image)

        # Iterate over contours to find and keep squares
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 sides and is convex
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                # Filter by area to avoid very small or large shapes
                if 10000 <= area <= 1000000:
                    # Fill the detected square on the mask
                    pass
                cv2.drawContours(mask, [approx], -1, (255, 255, 255), 150)

        # Apply the mask to the original frame
        cv_image = cv2.bitwise_and(cv_image, mask)
        # Show the result
        cv2.imshow("squares", cv_image)

if __name__ == "__main__":
    
    #predict()
    get_biggest_square()