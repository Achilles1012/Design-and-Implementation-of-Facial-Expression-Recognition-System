from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the face detector and emotion classifier
face_classifier = cv2.CascadeClassifier(r'C:\Users\Achilles2000\Desktop\Summer Internship\Testing\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\Achilles2000\Desktop\Summer Internship\Datasets\AffectNet\train_set\CNN_model.keras')

# Define the emotion labels
emotion_labels = ['Neutral','Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)
v_path=r'C:\Users\Achilles2000\Desktop\Summer Internship\Datasets\videoplayback.mp4'
cap = cv2.VideoCapture(v_path)
#cap = cv2.VideoCapture('http://10.11.0.105:8080/video')

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = r'C:\Users\Achilles2000\Desktop\Summer Internship\output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Warning: No frame captured, skipping this frame.")
        break

    # Convert the frame to grayscale (required by the face detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Extract the region of interest (ROI) corresponding to the face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the ROI to 48x48 pixels (required by the emotion classifier)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Preprocess the ROI for the emotion classifier
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict the emotion of the face
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            
            # Display the emotion label on the frame
            label_position = (x, y+h+30)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display "No Faces" if no face is detected
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write the frame to the output video
    out.write(frame)
    
    # Show the frame with the detected faces and emotions
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()
