import cv2


CAM_NUM = 0
WINDOW_NAME = "Camera"

cap = cv2.VideoCapture(CAM_NUM)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3))
# print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1))
print(cap.set(cv2.CAP_PROP_EXPOSURE, 0.001))

print(cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")))

print(cap.set(cv2.CAP_PROP_FPS, 30))
print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))
print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720))

while True:
    # Use a timer to keep track of speed of algorithm
    timer = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    fps = round(cv2.getTickFrequency() / (cv2.getTickCount() - timer), 2)

    window_title = f"Camera {CAM_NUM} | Resolution {frame.shape} | Frame rate {fps}"

    cv2.imshow(WINDOW_NAME, frame)
    cv2.setWindowTitle(WINDOW_NAME, window_title)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
