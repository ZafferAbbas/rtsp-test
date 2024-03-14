import cv2
import numpy as np
import face_recognition
import random
import string

def generate_code_word(length=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

cap = cv2.VideoCapture(0)

face_trackers = {}

frame_count = 0
unrecognized_frames_threshold = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)

    current_faces = {}

    for face_location in face_locations:
        top, right, bottom, left = face_location
        center_current = ((top + bottom) // 2, (left + right) // 2)

        matched_id = None

        for face_id, tracker_info in face_trackers.items():
            prev_location, _ = tracker_info
            center_prev = ((prev_location[0] + prev_location[2]) // 2, (prev_location[3] + prev_location[1]) // 2)
            distance = np.linalg.norm(np.array(center_current) - np.array(center_prev))

            if distance < 50:
                matched_id = face_id
                break

        if matched_id is None:
            new_id = generate_code_word()
            face_trackers[new_id] = (face_location, 0)
            current_faces[new_id] = face_location
        else:
            current_faces[matched_id] = face_location
            face_trackers[matched_id] = (face_location, 0)

    for face_id in list(face_trackers.keys()):
        if face_id not in current_faces:
            _, unrecognized_frames = face_trackers[face_id]
            unrecognized_frames += 1

            if unrecognized_frames >= unrecognized_frames_threshold:
                del face_trackers[face_id]
            else:
                face_trackers[face_id] = (face_trackers[face_id][0], unrecognized_frames)

    for face_id, (face_location, _) in face_trackers.items():
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_id, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
