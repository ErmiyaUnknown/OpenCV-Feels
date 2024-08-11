import cv2
from deepface import DeepFace

def get_age_and_emotion(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['age', 'emotion'])
        age = analysis[0]['age']
        emotion = analysis[0]['dominant_emotion']
        return age, emotion
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None, None

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # # Save the current frame as an image file
        # image_path = "temp_frame.jpg"
        # cv2.imwrite(image_path, frame)
        
        # Analyze the saved image
        age, emotion = get_age_and_emotion(frame)
        if age is not None and emotion is not None:
            # Draw a rectangle around the face (for demonstration, using a fixed rectangle here)
            height, width, _ = frame.shape
            cv2.rectangle(frame, (50, 50), (width - 50, height - 50), (255, 0, 0), 2)
            
            # Add text to the frame
            text = f"Age: {age}, Emotion: {emotion}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
