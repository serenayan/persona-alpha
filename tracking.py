# import cv2
# import dlib

# # Initialize dlib's face detector and then create the facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Start the webcam feed
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = detector(gray, 0)

#     # Loop over the faces
#     for face in faces:
#         # Determine the facial landmarks for the face region
#         shape = predictor(gray, face)
        
#         # Convert the facial landmark (x, y)-coordinates to a NumPy array
#         for i in range(0, shape.num_parts):
#             # Draw each landmark as a circle
#             cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), -1)

#     # Display the resulting frame
#     cv2.imshow("Frame", frame)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Start the webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and track the face, hands, and body
        results = holistic.process(image)

        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the face, hand, and pose annotations on the image
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Display the resulting image
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
