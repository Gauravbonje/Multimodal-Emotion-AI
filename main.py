import os
# 1. CONFIGURATION
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import threading
import queue
import numpy as np
import speech_recognition as sr
import torch
import csv
from datetime import datetime

from deepface import DeepFace
from transformers import pipeline

# --- HARDWARE ACCELERATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 System Running on: {device.upper()}")

# --- GLOBAL VARIABLES ---
current_video_emotion = "neutral"
current_audio_emotion = "listening..."
current_text_emotion = "waiting..."
current_text_content = ""

# --- QUEUES ---
audio_queue = queue.Queue()
video_queue = queue.Queue(maxsize=1) 

# --- CLASS: EMOTION SMOOTHER ---
class EmotionSmoother:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.current_probs = None 

    def update(self, new_probs):
        if self.current_probs is None:
            self.current_probs = new_probs
        else:
            for key in new_probs:
                if key in self.current_probs:
                    self.current_probs[key] = (self.current_probs[key] * (1 - self.alpha)) + (new_probs[key] * self.alpha)
        return max(self.current_probs, key=self.current_probs.get)

video_smoother = EmotionSmoother(alpha=0.4)

# --- LOAD MODELS ---
print("⏳ Loading AI Models...")
text_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1, device=device)
audio_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
print("✅ Models Loaded.")

# --- WORKER 1: VISUAL PROCESSOR ---
def visual_processor():
    global current_video_emotion
    while True:
        frame = video_queue.get()
        if frame is None: break
        try:
            objs = DeepFace.analyze(
                img_path=frame, 
                actions=['emotion'], 
                enforce_detection=False, 
                silent=True, 
                detector_backend='ssd' 
            )
            if objs:
                raw_probs = objs[0]['emotion']
                current_video_emotion = video_smoother.update(raw_probs)
        except: pass

# --- HELPER: SAFE LABEL EXTRACTOR (THE FIX) ---
def get_safe_label(prediction):
    try:
        # Case 1: It's a list of dicts [{'label': 'happy'}]
        if isinstance(prediction, list):
            # Case 1b: It's a Nested List [[{'label': 'happy'}]] (This was your error)
            if isinstance(prediction[0], list):
                return prediction[0][0]['label']
            return prediction[0]['label']
        # Case 2: It's just a dict {'label': 'happy'}
        elif isinstance(prediction, dict):
            return prediction['label']
        return "unknown"
    except Exception as e:
        return "error"

# --- WORKER 2: AUDIO PROCESSOR (ROBUST) ---
def audio_processor():
    global current_audio_emotion, current_text_emotion, current_text_content
    recognizer = sr.Recognizer()
    
    while True:
        audio_data = audio_queue.get()
        if audio_data == "STOP": break
        
        try:
            # 1. TRANSCRIPTION (Google Hinglish)
            text = recognizer.recognize_google(audio_data, language='en-IN')
            current_text_content = text
            
            if len(text) > 0:
                # 2. TEXT ANALYSIS
                text_pred = text_classifier(text)
                current_text_emotion = get_safe_label(text_pred) # <--- USE SAFE EXTRACTOR
            
            # 3. AUDIO ANALYSIS
            raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
            speech_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            audio_pred = audio_classifier(speech_array)
            current_audio_emotion = get_safe_label(audio_pred) # <--- USE SAFE EXTRACTOR
            
        except sr.UnknownValueError:
            pass 
        except Exception as e:
            # Print detailed error only if it's NOT the known list error (which is now fixed)
            print(f"AI Warning: {e}")

# --- WORKER 3: RECORDER ---
def recorder():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(sample_rate=16000) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎤 Mic Active (Google Mode).")
            while True:
                try:
                    audio = recognizer.listen(source, phrase_time_limit=4, timeout=None)
                    audio_queue.put(audio)
                except: continue
    except Exception as e:
        print(f"Mic Error: {e}")

# --- UI HELPER ---
def draw_ui(frame):
    cv2.rectangle(frame, (0, 0), (320, 180), (20, 20, 20), -1)
    cv2.rectangle(frame, (0, 0), (320, 180), (100, 100, 100), 1)

    cv2.putText(frame, "ACCURATE MODE (SSD)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    y = 60
    data = [("VISUAL", current_video_emotion, (0, 255, 0)), 
            ("AUDIO", current_audio_emotion, (0, 200, 255)), 
            ("TEXT", current_text_emotion, (255, 0, 255))]
            
    for label, val, col in data:
        cv2.putText(frame, f"{label}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, val.upper(), (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        y += 30
        
    txt = (current_text_content[:35] + '..') if len(current_text_content) > 35 else current_text_content
    cv2.putText(frame, f"'{txt}'", (20, 160), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255), 1)

# --- MAIN ---
def main():
    threading.Thread(target=recorder, daemon=True).start()
    threading.Thread(target=audio_processor, daemon=True).start()
    threading.Thread(target=visual_processor, daemon=True).start()

    cap = cv2.VideoCapture(0)
    print("🎥 System Live. Press 'Q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if video_queue.empty():
            video_queue.put(frame.copy())

        draw_ui(frame)
        cv2.imshow('High Accuracy Emotion AI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()