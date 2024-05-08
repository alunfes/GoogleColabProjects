import pyautogui
import cv2
import numpy as np
import time


# Define the coordinates of the area you want to capture
x1, y1 = 100, 100  # Top-left corner
x2, y2 = 500, 500  # Bottom-right corner

# Create a window to display the captured screenshots
cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)

# Continuously capture the screen image of the specific area every 1 second
while True:
    # Capture the screen image of the specific area
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    
    # Convert the screenshot to an OpenCV image
    screenshot_cv2 = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Display the screenshot in a window
    cv2.imshow("Screenshot", screenshot_cv2)
    
    # Wait for 1 second before capturing the next screenshot
    time.sleep(1)
    
    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows when the program ends
cv2.destroyAllWindows()
