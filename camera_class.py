from ultralytics import YOLO
import numpy as np
import rospy
import cv2


class CameraClass:

    def __init__(self, image):
        self.image = image
        # init node
        rospy.init_node("camera_class", anonymous=True)
        
        # center of the image
        self.centes = [856//2, 480//2]
        # biggest square
        self.biggest = 0

        # publisher to the camera node
        self.pub = rospy.Publisher("")

    
    def main(self):
        """
        Main function that computes the image and 
        sends the needed action from the drone.
        Steps to follow:
        - Predict the squares using YOLO
        - Get the biggest square from the frame
        - Get the center of that square
        - Return the command
        - Delete the image from the memory
        """
        # predict the image using YOLO
        self.predict()
        
        # get the biggest and cente square from the image
        self.get_biggest_center()

        # delete the image


    def predict(self):
        """
        Function to predict the images using YOLOv8
        Image saved is "image0"
        """
        # Load the trained model
        model = YOLO("/home/buly/Desktop/drone_competition/best.pt")

        # predict the image 
        model.predict(source=self.image, save=True)

        # update the image
        self.image = cv2.imread("/home/buly/Desktop/drone_competition/runs/detect/predict/image0.jpg")

    
    def get_biggest_square(self):
        """
        Function to get the biggest square from the frame
        """
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the same size as the self.cv_image, initialized to black
        mask = np.zeros_like(self.cv_image)

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
                    cv2.drawContours(mask, [approx], -1, (255, 255, 255), 150)

        # Apply the mask to the original frame
        self.cv_image = cv2.bitwise_and(self.cv_image, mask)
        # Show the result
        #cv2.imshow("squares", self.cv_image)


    

        