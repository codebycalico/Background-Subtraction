# for adaptation to Unity, see documentation:
# https://github.com/InstytutXR/OpenCV-plus-Unity/blob/master/source/unity/opencv-sharp/OpenCvSharp/modules/video/BackgroundSubtractorMog2.cs
# https://github.com/InstytutXR/OpenCV-plus-Unity/blob/master/source/unity/opencv-sharp/OpenCvSharp/modules/video/BackgroundSubtractor.cs
# Hough Lines from:
# https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# history is to adapt itself to the last x number of frames
subtractor = cv2.createBackgroundSubtractorMOG2(history=25, detectShadows=True)

while True:
    _, frame = cap.read()

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    mask = subtractor.apply(frame)

    edges = cv2.Canny(mask, 50, 100)

    # Hough Lines 
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_frame = np.copy(frame)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
            min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_frame,(x1,y1),(x2,y2),(255,0,0),5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(frame, 0.8, line_frame, 1, 0)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Line Frame", line_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()