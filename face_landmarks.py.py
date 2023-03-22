import mediapipe as mp
import cv2

# We are capturing video from our system camera i.e. 
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5
) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        
        results = face_mesh.process(image)

        for face_landmarks in results.multi_face_landmarks:
            print(face_landmarks)

        if not success:
            break
        cv2.imshow("My video capture", image)

        if cv2.waitKey(100)& 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



