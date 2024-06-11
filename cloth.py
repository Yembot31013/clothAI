import cv2
import mediapipe as mp
import requests
import PIL.Image
import google.generativeai as genai

genai.configure(api_key='')

model = genai.GenerativeModel('gemini-pro-vision')

def locate_body_landmarks(image):
  # Use Google Mediapipe to locate key body landmarks such as shoulders, waist, hips, etc.
  # Your code here
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  with mp_pose.Pose(
          static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("annoted image", annotated_image)
    cv2.waitKey(1)
    cv2.imwrite('annotated_image.jpg', annotated_image)
  landmarks = results.pose_landmarks
  return landmarks, annotated_image

def calculate_landmark_distance(landmarks):
  # Calculate distances between the landmarks from mediapipe to feed a machine learning model to estimate clothing measurements
  # Your code here
  mp_pose = mp.solutions.pose

  left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
  right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
  left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
  right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
  left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
  right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
  left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
  right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
  chest_circumference = abs(left_shoulder[0] - right_shoulder[0])
  waist_circumference = abs(left_hip[0] - right_hip[0])
  hip_circumference = abs(left_wrist[0] - right_wrist[0])
  inseam = abs(left_ankle[1] - right_ankle[1])
  sleeve_length = abs(left_wrist[1] - right_wrist[1])
  neck_circumference = abs(
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x - landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x)
  head_circumference = abs(
    landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y - landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y)

  # Convert measurements to inches
  chest_circumference *= 0.393701
  waist_circumference *= 0.393701
  hip_circumference *= 0.393701
  inseam *= 0.393701
  sleeve_length *= 0.393701
  neck_circumference *= 0.393701
  head_circumference *= 0.393701
  left_shoulder = tuple(x * 0.393701 for x in left_shoulder)
  right_shoulder = tuple(x * 0.393701 for x in right_shoulder)
  left_hip = tuple(x * 0.393701 for x in left_hip)
  right_hip = tuple(x * 0.393701 for x in right_hip)
  left_wrist = tuple(x * 0.393701 for x in left_wrist)
  right_wrist = tuple(x * 0.393701 for x in right_wrist)
  left_ankle = tuple(x * 0.393701 for x in left_ankle)
  right_ankle = tuple(x * 0.393701 for x in right_ankle)

  # Return everything as a sentence so that it can be easily read by the user
  
  return f"The chest circumference is {chest_circumference} inches, the waist circumference is {waist_circumference} inches, the hip circumference is {hip_circumference} inches, the inseam is {inseam} inches, the sleeve length is {sleeve_length} inches, the neck circumference is {neck_circumference} inches, the head circumference is {head_circumference} inches, the left shoulder is {left_shoulder} inches, the right shoulder is {right_shoulder} inches, the left hip is {left_hip} inches, the right hip is {right_hip} inches, the left wrist is {left_wrist} inches, the right wrist is {right_wrist} inches, the left ankle is {left_ankle} inches, the right ankle is {right_ankle} inches."
  
def calculate_measurement(landmark_details, annotated_image, annote_image):
  # convert the image to pillow format
  annotated_image = PIL.Image.fromarray(annotated_image)
  original_image = PIL.Image.fromarray(annote_image)
  
  # Use the generative model to estimate the size of the person's clothing
  prompt = f"""Given the person image and landmark image with landmark distance measurements in inches to make it easier to get an accurate cloth measurement value. note use all information given to you to get the estimated full cloth measurement. your task is to provide clothing measurement that a tailor can make use of to sew cloth. 
  landmark distance measurement: {landmark_details}"""
  
  response = model.generate_content([prompt, annotated_image, original_image])
  
  return response.text

def use_frame_to_measurment(original_image):
  if original_image is not None:
      landmarks, annotated_image = locate_body_landmarks(original_image)
      if landmarks:
          # Assume calculate_measurements is implemented
          landmark_details = calculate_landmark_distance(landmarks)

          measurment = calculate_measurement(
              landmark_details, annotated_image, original_image)

          print(measurment)

      else:
          print("No body landmarks detected in the isolated image.")
  else:
      print("Failed to isolate the person for landmark detection.")

capture = cv2.VideoCapture(0)

# list available cameras


# original_image = cv2.imread('person_image.jpg')

while capture.isOpened():
    ret, frame = capture.read()
    
    if ret:
      temp_frame = frame.copy()
      
      cv2.imshow('processing', temp_frame)
      
      # type q to quit
      if cv2.waitKey(10) & 0xFF == ord('q'):
          print("analzing...")
          original_image = frame.copy()
          # put loading... on the screen
          cv2.putText(temp_frame, 'Analzing Image...', (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
          cv2.imshow('analzing', temp_frame)
          cv2.waitKey(2)
          use_frame_to_measurment(original_image)