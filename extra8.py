import cv2
import numpy as np
import pyautogui
import time

# Function to bring PowerPoint to the foreground
def bring_powerpoint_to_front():
    pyautogui.hotkey('alt', 'tab')  # Simulate Alt+Tab to bring PowerPoint to the foreground

# Open a connection to the camera (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

# Flag to track whether the PowerPoint application has been brought to the foreground
powerpoint_in_foreground = False

# Initial position of the orange object
prev_center_x = None

# Counter to track the number of frames where the object is stationary
stationary_counter = 0

# Threshold for considering the object stationary
stationary_threshold = 20

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect orange color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

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
            print("Left")
            # You can perform an action here if needed
            stationary_counter = 0  # Reset the stationary counter

        else:
            print("Right")
            # Bring PowerPoint to the front if not already done
            if not powerpoint_in_foreground:
                bring_powerpoint_to_front()
                powerpoint_in_foreground = True

            # Check if there's a previous position for comparison
            if prev_center_x is not None:
                # Calculate displacement
                displacement = center_x - prev_center_x

                # Define a threshold for displacement to trigger slide change
                displacement_threshold = 50  # Adjust as needed

                # Check if displacement exceeds the threshold and the object is not stationary
                if displacement > displacement_threshold and stationary_counter < stationary_threshold:
                    # Perform action to go to the next slide
                    pyautogui.press('right')  # Use pyautogui to simulate key press
                    stationary_counter = 0  # Reset the stationary counter
                    time.sleep(1)  # Introduce a delay to prevent rapid consecutive changes

            # Update the previous center position
            prev_center_x = center_x

    else:
        # Increment the stationary counter if no orange particles are detected
        stationary_counter += 1

        # Reset flags and previous position if the object is stationary for too long
        if stationary_counter >= stationary_threshold:
            powerpoint_in_foreground = False
            prev_center_x = None

    # Stack the original frame and the mask horizontally for display
    result_frame = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))

    # Display the combined frame and mask
    cv2.imshow("Camera Feed with Mask", result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
