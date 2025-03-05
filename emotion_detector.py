import cv2
from deepface import DeepFace

# Webcam initialize karo
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam nahi khul rahi. Check karo!")
    exit()

# Webcam resolution set karo (performance ke liye)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Emotion Detector chal raha hai... 'q' press karo band karne ke liye.")

while True:
    # Frame capture karo
    ret, frame = cap.read()
    if not ret:
        print("Error: Video frame nahi mil raha.")
        break

    try:
        # Emotion detect karo using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]

        # Poore frame ke liye ek border rectangle
        h, w, _ = frame.shape
        cv2.rectangle(frame, (20, 20), (w-20, h-20), (0, 255, 0), 2)

        # Dominant emotion aur confidence screen pe dikhao
        text = f"Dominant Emotion: {emotion} ({confidence:.2f}%)"
        cv2.putText(frame, text, (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 0, 0), 2)

        # Saare emotions ke scores bhi dikhao
        emotions = result[0]['emotion']
        y_offset = 40
        for emotion_name, score in emotions.items():
            emotion_text = f"{emotion_name}: {score:.2f}%"
            cv2.putText(frame, emotion_text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20

        # Suggestion add karo based on emotion
        if emotion == "sad":
            cv2.putText(frame, "Suggestion: Listen to some music!", 
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif emotion == "angry":
            cv2.putText(frame, "Suggestion: Take a deep breath!", 
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        elif emotion == "happy":
            cv2.putText(frame, "Suggestion: Keep smiling!", 
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    except Exception as e:
        # Agar face detect nahi hota ya koi error aata hai
        cv2.putText(frame, "No face detected", (20, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Frame display karo
    cv2.imshow("Emotion Detector", frame)

    # 'q' press karne pe band karo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup karo
cap.release()
cv2.destroyAllWindows()
print("Emotion Detector band ho gaya.")