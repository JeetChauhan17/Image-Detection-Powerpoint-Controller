import cv2
import numpy as np
import pyautogui
import time

# init camera
cap = cv2.VideoCapture(0)

# check if ppt in  foreground
powerpoint_in_foreground = False

# Initialialize positions objects
prev_center_x_orange = None
prev_center_x_pink = None


stationary_counter_orange = 0
stationary_counter_pink = 0

# how long should it be stationary
stationary_threshold = 20

# check if cam inverted
invert_camera = True

while True:
    ret, frame = cap.read()
    
    # Flip cam if inverted
    if invert_camera:
        frame = cv2.flip(frame, 1)
    
    # Detect orange color
    hsv_orange = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])
    mask_orange = cv2.inRange(hsv_orange, lower_orange, upper_orange)
    
    # Detect pink color
    hsv_pink = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([150, 100, 100])
    upper_pink = np.array([170, 255, 255])
    mask_pink = cv2.inRange(hsv_pink, lower_pink, upper_pink)
    
    # Contours in masks
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process orange object
    if contours_orange:
        largest_contour_orange = max(contours_orange, key=cv2.contourArea)
        x_orange, y_orange, w_orange, h_orange = cv2.boundingRect(largest_contour_orange)
        center_x_orange = x_orange + w_orange // 2
        
        if center_x_orange < frame.shape[1] // 2:
            print("Left (Orange)")
            stationary_counter_orange = 0
        else:
            print("Right (Orange)")
            if prev_center_x_orange is not None:
                displacement_orange = center_x_orange - prev_center_x_orange
                displacement_threshold_orange = 50
                if displacement_orange > displacement_threshold_orange and stationary_counter_orange < stationary_threshold:
                    pyautogui.press('right')
                    stationary_counter_orange = 0
                    time.sleep(1)
            prev_center_x_orange = center_x_orange
    else:
        stationary_counter_orange += 1
        if stationary_counter_orange >= stationary_threshold:
            prev_center_x_orange = None
    
    # Process pink object
    if contours_pink:
        largest_contour_pink = max(contours_pink, key=cv2.contourArea)
        x_pink, y_pink, w_pink, h_pink = cv2.boundingRect(largest_contour_pink)
        center_x_pink = x_pink + w_pink // 2
        
        if center_x_pink < frame.shape[1] // 2:
            print("Left (Pink)")
            stationary_counter_pink = 0
        else:
            print("Right (Pink)")
            if prev_center_x_pink is not None:
                displacement_pink = center_x_pink - prev_center_x_pink
                displacement_threshold_pink = 50
                if displacement_pink > displacement_threshold_pink and stationary_counter_pink < stationary_threshold:
                    pyautogui.press('left')
                    stationary_counter_pink = 0
                    time.sleep(1)
            prev_center_x_pink = center_x_pink
    else:
        stationary_counter_pink += 1
        if stationary_counter_pink >= stationary_threshold:
            powerpoint_in_foreground = False
            prev_center_x_pink = None
    
    # combine frame and masks
    result_frame = np.hstack((frame, cv2.cvtColor(mask_orange, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_pink, cv2.COLOR_GRAY2BGR)))
    
    # Display res frame
    cv2.imshow("Camera Feed with Masks", result_frame)
    
    # wxit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
