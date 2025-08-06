from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
app=Flask(__name__)
camera = cv2.VideoCapture(0)


Avishek_image = face_recognition.load_image_file("Avishek/Avishek.jpg")
Avishek_face_encoding = face_recognition.face_encodings(Avishek_image)[0]

Swarnajyoti_image = face_recognition.load_image_file("Swarnajyoti/Swarnajyoti.jpg")
Swarnajyoti_face_encoding = face_recognition.face_encodings(Swarnajyoti_image)[0]

Priya_image = face_recognition.load_image_file("Priya/Priya.jpg")
Priya_face_encoding = face_recognition.face_encodings(Priya_image)[0]

Kalpatru_image = face_recognition.load_image_file("Kalpatru/Kalpatru.jpg")
Kalpatru_face_encoding = face_recognition.face_encodings(Kalpatru_image)[0]

Biprarshi_image = face_recognition.load_image_file("Biprarshi/Biprarshi.jpg")
Biprarshi_face_encoding = face_recognition.face_encodings(Biprarshi_image)[0]

Disha_image = face_recognition.load_image_file("Disha/Disha.jpg")
Disha_face_encoding = face_recognition.face_encodings(Disha_image)[0]

Subhas_sir_image = face_recognition.load_image_file("Subhas_sir/Subhas_sir.jpg")
Subhas_sir_encoding = face_recognition.face_encodings(Subhas_sir_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    Avishek_face_encoding,
    Swarnajyoti_face_encoding,
    Priya_face_encoding,
    Kalpatru_face_encoding,
    Biprarshi_face_encoding,
    Disha_face_encoding,
    Subhas_sir_encoding,

]
known_face_names = [
    "Avishek|C.S-3|C.R=Murderer",
    "Swarnajyoti|C.S-0|C.R=None",
    "Priya|C.S-1|C.R=None",
    "Kalpatru|C.S-2|C.R=Bribery",
    "Biprarshi|C.S-3|C.R=Kidnapping",
    "Disha|C.S-1|C.R=None",
    "Mr Subhas|C.S-5|C.R-None"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)