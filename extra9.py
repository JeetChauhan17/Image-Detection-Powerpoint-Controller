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

# Initial positions of the orange and pink objects
prev_center_x_orange = None
prev_center_x_pink = None

# Counters to track the number of frames where the objects are stationary
stationary_counter_orange = 0
stationary_counter_pink = 0

# Threshold for considering the objects stationary
stationary_threshold = 20

# Camera inversion flag
invert_camera = True

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the camera is inverted, flip the frame
    if invert_camera:
        frame = cv2.flip(frame, 1)

    # Detect orange color
    hsv_orange = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_orange = np.array([0, 100, 100])
    # upper_orange = np.array([30, 255, 255])
    # Adjusted color ranges for orange
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask_orange = cv2.inRange(hsv_orange, lower_orange, upper_orange)

    # Detect pink color
    hsv_pink = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_pink = np.array([140, 100, 100])
    # upper_pink = np.array([170, 255, 255])
    lower_pink = np.array([150, 100, 100])
    upper_pink = np.array([170, 255, 255])

    mask_pink = cv2.inRange(hsv_pink, lower_pink, upper_pink)

    # Find contours in the orange mask
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_orange:
        # Get the largest orange contour
        largest_contour_orange = max(contours_orange, key=cv2.contourArea)

        # Get the bounding box of the largest orange contour
        x_orange, y_orange, w_orange, h_orange = cv2.boundingRect(largest_contour_orange)

        # Determine the center of the bounding box
        center_x_orange = x_orange + w_orange // 2

        # Draw a line to separate the screen into left and right
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 2)

        # Check if the center is on the left or right
        if center_x_orange < frame.shape[1] // 2:
            print("Left (Orange)")
            # You can perform an action here if needed
            stationary_counter_orange = 0  # Reset the stationary counter

        else:
            print("Right (Orange)")
            # Bring PowerPoint to the front if not already done
            if not powerpoint_in_foreground:
                bring_powerpoint_to_front()
                powerpoint_in_foreground = True

            # Check if there's a previous position for comparison
            if prev_center_x_orange is not None:
                # Calculate displacement
                displacement_orange = center_x_orange - prev_center_x_orange

                # Define a threshold for displacement to trigger slide change
                displacement_threshold_orange = 50  # Adjust as needed

                # Check if displacement exceeds the threshold and the object is not stationary
                if displacement_orange > displacement_threshold_orange and stationary_counter_orange < stationary_threshold:
                    # Perform action to go to the next slide
                    pyautogui.press('right')  # Use pyautogui to simulate key press
                    stationary_counter_orange = 0  # Reset the stationary counter
                    time.sleep(1)  # Introduce a delay to prevent rapid consecutive changes

            # Update the previous center position for the orange object
            prev_center_x_orange = center_x_orange

    else:
        # Increment the stationary counter if no orange particles are detected
        stationary_counter_orange += 1

        # Reset flags and previous position if the orange object is stationary for too long
        if stationary_counter_orange >= stationary_threshold:
            prev_center_x_orange = None

    # Find contours in the pink mask
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_pink:
        # Get the largest pink contour
        largest_contour_pink = max(contours_pink, key=cv2.contourArea)

        # Get the bounding box of the largest pink contour
        x_pink, y_pink, w_pink, h_pink = cv2.boundingRect(largest_contour_pink)

        # Determine the center of the bounding box
        center_x_pink = x_pink + w_pink // 2

        # Draw a line to separate the screen into left and right
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 2)

        # Check if the center is on the left or right
        if center_x_pink < frame.shape[1] // 2:
            print("Left (Pink)")
            # You can perform an action here if needed
            stationary_counter_pink = 0  # Reset the stationary counter

        else:
            print("Right (Pink)")
            # Bring PowerPoint to the front if not already done
            if not powerpoint_in_foreground:
                bring_powerpoint_to_front()
                powerpoint_in_foreground = True

            # Check if there's a previous position for comparison
            if prev_center_x_pink is not None:
                # Calculate displacement
                displacement_pink = center_x_pink - prev_center_x_pink

                # Define a threshold for displacement to trigger slide change
                displacement_threshold_pink = 50  # Adjust as needed

                # Check if displacement exceeds the threshold and the object is not stationary
                if displacement_pink > displacement_threshold_pink and stationary_counter_pink < stationary_threshold:
                    # Perform action to go to the previous slide
                    pyautogui.press('left')  # Use pyautogui to simulate key press
                    stationary_counter_pink = 0  # Reset the stationary counter
                    time.sleep(1)  # Introduce a delay to prevent rapid consecutive changes

            # Update the previous center position for the pink object
            prev_center_x_pink = center_x_pink

    else:
        # Increment the stationary counter if no pink particles are detected
        stationary_counter_pink += 1

        # Reset flags and previous position if the pink object is stationary for too long
        if stationary_counter_pink >= stationary_threshold:
            powerpoint_in_foreground = False
            prev_center_x_pink = None

    # Stack the original frame, orange mask, and pink mask horizontally for display
    result_frame = np.hstack((frame, cv2.cvtColor(mask_orange, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_pink, cv2.COLOR_GRAY2BGR)))

    # Display the combined frame and masks
    cv2.imshow("Camera Feed with Masks", result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
