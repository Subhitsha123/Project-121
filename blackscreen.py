from os import access
import cv2 
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
capture = cv2.VideoCapture(0)
img = cv2.imread("me.jpeg") 
time.sleep(2)
bg = 0

for i in range(60):
    ret, bg = capture.read()

bg = np.flip(bg, axis=1)

while(capture.isOpened()):
    ret, img  = capture.read()

    if not ret :
        break

    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_black = np.array([30, 30, 0])
    u_black = np.array([104, 153, 70])
    mask_1 = cv2.inRange(hsv, l_black, u_black)

    l_black = np.array([170, 120, 70])
    u_black = np.array([180, 255, 255])
    mask_2 = cv2.inRange(hsv, l_black, u_black)

    mask_1 = mask_1 + mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    resolution_1 = cv2.bitwise_and(img, img, mask = mask_2)
    resolution_2 = cv2.bitwise_and(img, img, mask = mask_1)

    #Generating the final output by merging res_1 and res_2
    final_output = cv2.addWeighted(resolution_1, 1, resolution_2, 1, 0)
    output_file.write(final_output)
    #Displaying the output to the user
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)


capture.release()
cv2.destroyAllWindows()