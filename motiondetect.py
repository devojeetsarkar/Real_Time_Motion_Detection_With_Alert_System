import cv2
import numpy as np
import pygame
import os
from datetime import datetime

# Initialize Pygame mixer
pygame.mixer.init()

def play_sound():
    """Play the alert sound if the file exists."""
    alert_sound_path = 'C:/Users/amshu/Downloads/motionalarmdetection/alert.wav'
    if os.path.exists(alert_sound_path):
        pygame.mixer.music.load(alert_sound_path)
        pygame.mixer.music.play()
    else:
        print(f"Error: {alert_sound_path} file not found.")

def save_motion_image(frame):
    """Save a photo when motion is detected."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"motion_capture_{timestamp}.jpg"
    save_path = os.path.join('C:/Users/amshu/Downloads/motionalarmdetection/', filename)
    cv2.imwrite(save_path, frame)
    print(f"Motion detected. Image saved as {filename}")

def main():
    """Main function to capture video, detect motion, and play sound."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Create a background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    is_fullscreen = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)
        
        # Filter out noise and small objects using morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 1500:  # Adjust the threshold as needed
                continue

            # Draw a rectangle around the detected contour
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

        # Play sound alert and save photo if motion is detected
        if motion_detected:
            play_sound()
            save_motion_image(frame)

        # Show the frame with detected motion
        cv2.imshow("Motion Detection", frame)
        
        # Handle key events
        key = cv2.waitKey(10)
        if key == 27:  # ESC key
            break
        elif key == ord('f'):  # 'f' key to toggle full screen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty("Motion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Motion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


