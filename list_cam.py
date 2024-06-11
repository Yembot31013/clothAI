import cv2

# List available cameras
camera_list = []
for i in range(10):
  cap = cv2.VideoCapture(i)
  if cap.isOpened():
    camera_list.append(i)
    _, frame = cap.read()
    cv2.imshow(f"Camera {i}", frame)

print("Available cameras:", camera_list)
