# Tri-Modal Fusion Network for Robust Emotion Recognition

## Overview
This project implements a Real-Time Multimodal Emotion Recognition System using State-of-the-Art (SOTA) pre-trained deep learning models. It simultaneously captures and analyzes three distinct modalities—Visual (Facial Expressions), Audio (Vocal Tone), and Text (Semantic Meaning)—to provide a comprehensive psychological profile of the user in real-time.



## Key Research Features
* **Asynchronous Multi-Threading:** Prevents I/O blocking by separating video capture, visual inference, and audio processing into isolated, parallel threads. Ensures a smooth video feed without freezing during neural network inference.
* **Exponential Moving Average (EMA) Smoothing:** Implements a mathematical `EmotionSmoother` class to eliminate "emotion jitter" (rapidly flickering predictions caused by micro-shadows or movement), resulting in stable, high-fidelity visual accuracy.
* **Apple Silicon (MPS) Acceleration:** Fully optimized for Mac M1/M2 chips, utilizing `torch.backends.mps` for local neural engine processing.
* **Hinglish Optimized ASR:** Uses Google's Speech-to-Text API configured for `en-IN` to accurately transcribe mixed Hindi/English accents in real-time.
* **Robust Error Handling:** Includes a custom `get_safe_label()` extractor to dynamically handle varied JSON/List pipeline outputs from Hugging Face models without crashing.

## System Architecture & Models

1. **Visual Pipeline (Face)**
   * **Framework:** DeepFace
   * **Detector Backend:** `ssd` (Single Shot Multibox Detector) for high accuracy across different head poses and lighting conditions.
   * **Model:** VGG-Face (Pre-trained CNN).

2. **Audio Pipeline (Voice Tone)**
   * **Framework:** Hugging Face Transformers
   * **Model:** `superb/wav2vec2-base-superb-er`
   * **Architecture:** Wav2Vec 2.0 (Transformer-based self-supervised learning directly on raw audio waveforms).

3. **Text Pipeline (Semantics)**
   * **Framework:** Hugging Face Transformers
   * **Model:** `j-hartmann/emotion-english-distilroberta-base`
   * **Architecture:** DistilRoBERTa (Fine-tuned on 6 diverse emotion datasets).

## Prerequisites
* **Operating System:** macOS (Optimized for Apple Silicon M1/M2/M3)
* **Python Version:** 3.10.x
* **Hardware:** Webcam and Microphone

## Installation Guide

1. **Clone/Setup the Repository:**
   Open your terminal and create a dedicated folder.
   ```bash
   mkdir EmotionProject
   cd EmotionProject

Create a Conda Environment:
We highly recommend Python 3.10 for dependency stability.

Bash
conda create -n emotion_env python=3.10 -y
conda activate emotion_env
Install System Audio Libraries:
Note: This requires Homebrew installed on your Mac.

Bash
brew install portaudio ffmpeg
Install the Optimized Dependencies:
Note: This specific combination prevents protobuf and TensorFlow-Metal conflicts on Mac M2.

Bash
pip install tensorflow-macos==2.16.2 tensorflow-metal==1.1.0 tf-keras==2.16.0 protobuf==3.20.3 h5py==3.10.0 deepface==0.0.95
pip install torch torchvision torchaudio transformers speechrecognition pyaudio opencv-python numpy
How to Run
Ensure your conda environment is activated:

Bash
conda activate emotion_env
Run the main script:

Bash
python main.py
Look at the camera and speak naturally. The UI will update in real-time.

Press q while the video window is in focus to safely terminate the system and close the background threads.

(Note: On the very first run, the system will download the pre-trained weights for the Hugging Face and SSD models. This may take a few minutes depending on your internet connection.)

Citations & Academic References
This project utilizes models and architectures proposed in the following research papers:

Visual: Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). "Deep Face Recognition". British Machine Vision Conference (BMVC).

Vision Backend (SSD): Liu, W., et al. (2016). "SSD: Single Shot MultiBox Detector". European Conference on Computer Vision (ECCV).

Audio: Baevski, A., et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations". Advances in Neural Information Processing Systems (NeurIPS).

Text: Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". arXiv preprint arXiv:1910.01108.
