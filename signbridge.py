import cv2
import anthropic
from elevenlabs.client import ElevenLabs
import time
import mediapipe as mp
import os
import azure.cognitiveservices.speech as speechsdk

ANTHROPIC_API_KEY = ""
ELEVENLABS_API_KEY = ""
AZURE_SPEECH_KEY = ""
AZURE_REGION = ""

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
el_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_buffer = []
last_time = time.time()

def ask_claude(words):
    prompt = f"""You are helping a deaf patient communicate with a doctor in an emergency.
The patient signed these words: {', '.join(words)}
Form a clear, concise medical sentence. Under 15 words. Only return the sentence."""
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def speak(text):
    audio = el_client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2"
    )
    filename = f"output_{int(time.time())}.mp3"
    with open(filename, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    from playsound import playsound
    playsound(filename)

def doctor_listen():
    config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    recognizer = speechsdk.SpeechRecognizer(speech_config=config)
    print("Doctor mode ON - speak now...")
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    return None

print("SignBridge ready!")
print("SPACE=send to Claude | C=clear | D=doctor mode | Q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            tips = [8, 12, 16, 20]
            fingers_up = sum(1 for tip in tips if hand_lm.landmark[tip].y < hand_lm.landmark[tip-2].y)
            if fingers_up == 0:
                gesture = "no"
            elif fingers_up >= 3:
                gesture = "yes"
            elif fingers_up == 1:
                gesture = "pain"
            elif fingers_up == 2:
                gesture = "help"
            else:
                gesture = None
            if gesture and time.time() - last_time > 2:
                gesture_buffer.append(gesture)
                last_time = time.time()
                print(f"Detected: {gesture}")

    cv2.putText(frame, f"Words: {' '.join(gesture_buffer)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Fist=no | 1finger=pain | 2fingers=help | 5fingers=yes", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "SPACE=Claude | C=Clear | D=Doctor | Q=Quit", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("SignBridge", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        gesture_buffer = []
        print("Cleared!")
    elif key == ord('d'):
        doctor_text = doctor_listen()
        if doctor_text:
            print(f"Doctor said: {doctor_text}")
            result_frame = frame.copy()
            cv2.putText(result_frame, f"DR: {doctor_text}", (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.imshow("SignBridge", result_frame)
            cv2.waitKey(4000)
    elif key == ord(' ') and gesture_buffer:
        print("Sending to Claude...")
        sentence = ask_claude(gesture_buffer)
        print(f"Claude says: {sentence}")
        speak(sentence)
        gesture_buffer = []
        result_frame = frame.copy()
        cv2.putText(result_frame, sentence, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("SignBridge", result_frame)
        cv2.waitKey(3000)

cap.release()
cv2.destroyAllWindows()