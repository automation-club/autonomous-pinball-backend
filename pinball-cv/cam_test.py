# import cv2
#
#
# CAM_NUM = 6
# WINDOW_NAME = "Camera"
#
# cap = cv2.VideoCapture(CAM_NUM)
# # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
#
# # print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3))
# # print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1))
# # print(cap.set(cv2.CAP_PROP_EXPOSURE, 0.001))
#
# # print(cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")))
# #
# # print(cap.set(cv2.CAP_PROP_FPS, 30))
# # print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))
# # print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720))
#
# while True:
#     # Use a timer to keep track of speed of algorithm
#     timer = cv2.getTickCount()
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     fps = round(cv2.getTickFrequency() / (cv2.getTickCount() - timer), 2)
#
#     window_title = f"Camera {CAM_NUM} | Resolution {frame.shape} | Frame rate {fps}"
#
#     cv2.imshow(WINDOW_NAME, frame)
#     cv2.setWindowTitle(WINDOW_NAME, window_title)
#
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2



"""
Test the ports and returns a tuple with the available ports and the ones that are working.
"""
is_working = True
dev_port = 0
working_ports = []
available_ports = []
while is_working:
    camera = cv2.VideoCapture(dev_port)
    if not camera.isOpened():
        is_working = False
        print("Port %s is not working." %dev_port)
    else:
        is_reading, img = camera.read()
        w = camera.get(3)
        h = camera.get(4)
        if is_reading:
            print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
            working_ports.append(dev_port)
        else:
            print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
            available_ports.append(dev_port)
    dev_port +=1

print(available_ports)
print(working_ports)
