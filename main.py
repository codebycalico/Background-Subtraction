import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# history is to adapt itself to the last x number of frames
subtractor = cv2.createBackgroundSubtractorMOG2(history=25, detectShadows=True)

while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()