import HandTrackingModule as htm
import cv2
import numpy as np
import time
import os

folder = "Header"
video = cv2.VideoCapture(0)
list = os.listdir("Header")
overlayList=[]
video.set(3,1280)
video.set(4,720)
drawColor=(255,0,255)
for i in list:
    image = cv2.imread(f'{folder}/{i}')
    overlayList.append(image)
header = overlayList[0]
xp,yp=0,0
detector = htm.handDetector(detectionCon=0.85,maxHands=1 )
imgcanvas=np.zeros((720,1280,3),np.uint8)
while True:
    success, img = video.read()
    img = cv2.flip(img, 1)

    # Detect hand and find landmarks
    img = detector.findHands(img)
    pos = detector.findPosition(img)

    if len(pos) != 0:
        # Get the tip positions of the index and middle fingers
        x1, y1 = pos[8][1:]  # Index finger tip
        x2, y2 = pos[12][1:]  # Middle finger tip

        # Check which fingers are up
        up = detector.fingersUp()

        # Selection mode (two fingers up)
        if up[1] and up[2]:
            xp, yp = 0, 0  # Reset the previous points
            print("Selection mode")
            if y1 < 140:
                # Check for color selection
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[10]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[5]
                    drawColor = (0, 0, 0)

        # Drawing mode (index finger up)
        # Drawing mode (index finger up)
        if up[1] and not up[2]:
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1  # Initialize previous points

            if drawColor == (0, 0, 0):  # Eraser mode
                thickness = 55  # Increased thickness for eraser
            else:
                thickness = 15  # Normal thickness for drawing

            # Draw a line on the canvas
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgcanvas, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1  # Update previous points

    # Merge the canvas with the original image
    grayCanvas = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(grayCanvas, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgcanvas)

    # Add the header image to the frame
    img[0:250, 0:1280] = header

    # Display the final output
    cv2.imshow("Video", img)
    cv2.imshow("Canvas", imgcanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
