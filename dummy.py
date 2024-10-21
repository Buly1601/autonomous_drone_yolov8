import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
#import tensorflow as tf


class BebopCameraDisplay:


    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('bebop_camera_display', anonymous=True)

        # Initialize the CvBridge object
        self.bridge = CvBridge()

        # get the squares in frame
        self.square = None
        # get biggest area
        self.biggest = 0
        # upper and lower color range
        self.mask_defined = False

        # center of camera
        self.center = (856//2, 480//2)

        # Subscribe to the Bebop camera image topic
        self.image_sub = rospy.Subscriber("/bebop/image_raw", Image, self.image_callback)
        
        # Create a named window for displaying the video
        cv2.namedWindow("Bebop Camera", cv2.WINDOW_NORMAL)
        
        # Spin to keep the node running
        rospy.spin()

    def define_masks(self):
        # Define HSV ranges
        self.lower_red = np.array([0, 100, 100])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_orange = np.array([10, 100, 100])
        self.upper_orange = np.array([30, 255, 255])

        # Create masks for red and orange
        self.mask_red1 = cv2.inRange(self.hsv, self.lower_red, self.upper_red)
        self.mask_red2 = cv2.inRange(self.hsv, self.lower_red2, self.upper_red2)
        self.mask_orange = cv2.inRange(self.hsv, self.lower_orange, self.upper_orange)
    
        # Combine the red masks
        self.mask_red = self.mask_red1 | self.mask_red2
        # Combine red and orange masks
        self.mask_combined = self.mask_red | self.mask_orange

        # just do it once
        self.mask_defined = True


    def image_callback(self, msg):
        """
        Callback function to handle image messages from the drone's camera.
        """
        try:
            # Convert the ROS Image message to an OpenCV format
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detect_and_keep_squares()
            
            self.hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            if not self.mask_defined:
                self.define_masks()
            # get colored image
            self.only_orange()
            # get squares
            self.find_squares() 
            
            
            # Wait for a key press, with a short delay to allow OpenCV to refresh the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Exit if 'q' is pressed
                rospy.signal_shutdown("User exit")
        
        except Exception as e:
            rospy.logerr("Error converting or displaying image: %s", e)


    def shutdown(self):
        """
        Properly shutdown the OpenCV window when the node is stopped.
        """
        cv2.destroyAllWindows()


    def detect_and_keep_squares(self):
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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


    def only_orange(self):
        """
        Get only orangy colors from image
        """
        # Create masks for red and orange
        mask_red1 = cv2.inRange(self.hsv, self.lower_red, self.upper_red)
        mask_red2 = cv2.inRange(self.hsv, self.lower_red2, self.upper_red2)
        mask_orange = cv2.inRange(self.hsv, self.lower_orange, self.upper_orange)
        # Combine the red masks
        mask_red = mask_red1 | mask_red2

        # Combine red and orange masks
        mask_combined = mask_red | mask_orange
        self.cv_image = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask_combined)

    
    def find_squares(self):
        "Finds squares in an image"
        # apply gaussian blur to reduce noise
        self.blured = cv2.GaussianBlur(self.cv_image, (5,5), 0)
        # perform canny edge detection
        self.edges = cv2.Canny(self.blured, 50, 150)
        # find contours
        contours, _ = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the contour has 4 vertices
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Check the aspect ratio to determine if it's a square
                _, _, width, height = cv2.boundingRect(approx)
                aspect_ratio = float(width) / height

                # Allow imperfections in the aspect ratio
                if 0.9 <= aspect_ratio <= 1.1:
                    # Get the area to identify the largest square
                    area = cv2.contourArea(approx)

                    
                    # Check if the current square is the largest found
                    if area > self.biggest * 0.8:
                        self.biggest = area
                        self.square = approx
                        # Calculate the center of the self.square
                        M = cv2.moments(self.square)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0  # Avoid division by zero
                        
                        # Draw a red point at the center of the square
                        cv2.circle(self.cv_image, (cX, cY), 5, (0, 0, 255), -1)  
                        cv2.drawContours(self.cv_image, [self.square], -1, (0, 255, 0), 2) 
        
        # draw center
        self.img = cv2.circle(self.cv_image, self.center, 2, (255, 0, 0), 2)
        # draw error circle
        self.img = cv2.circle(self.cv_image, self.center, 20, (255, 0, 0), 1) 

        # Show the result
        cv2.imshow("squares", self.cv_image)


if __name__ == '__main__':
    try:
        # Create and run the Bebop camera display node
        display = BebopCameraDisplay()
        
    except rospy.ROSInterruptException:
        pass
