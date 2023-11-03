import cv2
import numpy as np
import time
import pyautogui  # Added for PowerPoint control

# Global variables
left_timer = 0
right_timer = 0
threshold_time = 7  # in seconds
decision_made = False
start_time = None

def preprocess_image(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv

def filter_orange_pixels(hsv):
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result, mask

def detect_orange_color(frame):
    global left_timer, right_timer, decision_made, start_time

    hsv = preprocess_image(frame)
    orange_mask, mask = filter_orange_pixels(hsv)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Determine the center of the bounding box
        center_x = x + w // 2

        # Draw a line to separate the screen into left and right
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 2)

        # Check if the center is on the left or right
        if center_x < frame.shape[1] // 2:
            left_timer += 1
            right_timer = 0
        else:
            right_timer += 1
            left_timer = 0

        # Start the decision timer if not already started
        if not decision_made:
            start_time = time.time()
            decision_made = True

        # Check if the timer exceeds the threshold
        elapsed_time = time.time() - start_time
        if elapsed_time >= threshold_time:
            if left_timer >= right_timer:
                print("Left")
                # Perform action to go to the previous slide (if needed)
            else:
                print("Right")
                # Perform action to go to the next slide
                pyautogui.press('right')  # Use pyautogui to simulate key press
            decision_made = False
            start_time = None

    else:
        # Reset the decision timer if no orange particles are detected
        decision_made = False
        start_time = None

    # Stack the original frame and the mask horizontally for display
    result_frame = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))

    return result_frame

# Open a connection to the camera (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect orange color and control PowerPoint
    result_frame = detect_orange_color(frame)

    # Display the combined frame and mask
    cv2.imshow("Camera Feed with Mask", result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
