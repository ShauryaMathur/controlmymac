import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import queue
from math import hypot
import pyaudio
import librosa
import librosa.display
import sounddevice as sd
from IPython.core.display_functions import display
from scipy import signal
from scipy.signal import find_peaks
import threading
import time
from IPython.display import Audio
import csv
import matplotlib.pyplot as plt

class AdvancedSnapDetector:
    def __init__(self, callback, threshold=0.3):
        self.callback = callback
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000  # Standard rate for yamnet
        self.THRESHOLD = threshold
        self.audio_queue = queue.Queue()
        self.is_running = True

        # Audio analysis parameters
        self.hop_length = 512
        self.n_mels = 128
        self.fmin = 20
        self.fmax = 8000

        # Load the yamnet model
        self.model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
        self.class_map_path = self.model.class_map_path().numpy()
        self.class_names = self.class_names_from_csv(self.class_map_path)

    # Find the name of the class with the top score when mean-aggregated across frames.
    def class_names_from_csv(self,class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        class_names = []
        with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])

        return class_names

    def start(self):
        self.audio_thread = threading.Thread(target=self._audio_process_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()

    def stop(self):
        self.is_running = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        # Add noise gate
        noise_threshold = 0.02
        if np.max(np.abs(audio_data)) > noise_threshold:
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def _analyze_audio_features(self, audio_data):
        """Advanced audio analysis using librosa"""
        try:
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)

            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            print(audio_data)

            # Calculate features
            # 1. Spectral Centroid
            # cent = librosa.feature.spectral_centroid(y=audio_data, sr=self.RATE)[0]
            #
            # # 2. Onset Strength
            # onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.RATE)
            #
            # # 3. Zero Crossing Rate
            # zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            #
            # # 4. MFCC for timbral characteristics
            # mfcc = librosa.feature.mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)
            # print('Values',np.mean(cent),np.max(onset_env),np.mean(zcr),np.std(mfcc[0]))
            # # Define snap characteristics
            # is_snap = (
            #         np.mean(cent) > 2000 and  # High spectral centroid
            #         np.max(onset_env) > self.THRESHOLD and  # Sharp onset
            #         np.mean(zcr) > 0.05 and  # High zero crossing rate
            #         np.std(mfcc[0]) > 0.5  # Variable timbral characteristics
            # )

            # sd.play(audio_data, self.RATE)
            # sd.wait()

            # Run the model, check the output.
            scores, embeddings, spectrogram = self.model(audio_data)
            scores_np = scores.numpy()
            spectrogram_np = spectrogram.numpy()
            infered_class = self.class_names[scores_np.mean(axis=0).argmax()]
            print(f'The main sound is: {infered_class}')

            plt.figure(figsize=(10, 6))

            # Plot the waveform.
            plt.subplot(3, 1, 1)
            plt.plot(audio_data)
            plt.xlim([0, len(audio_data)])

            # Plot the log-mel spectrogram (returned by the model).
            plt.subplot(3, 1, 2)
            plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

            # Plot and label the model output scores for the top-scoring classes.
            mean_scores = np.mean(scores, axis=0)
            top_n = 10
            top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
            plt.subplot(3, 1, 3)
            plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

            # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
            # values from the model documentation
            patch_padding = (0.025 / 2) / 0.01
            plt.xlim([-patch_padding - 0.5, scores.shape[0] + patch_padding - 0.5])
            # Label the top_N classes.
            yticks = range(0, top_n, 1)
            plt.yticks(yticks, [self.class_names[top_class_indices[x]] for x in yticks])
            _ = plt.ylim(-0.5 + np.array([top_n, 0]))
            plt.show()
            return True if infered_class == 'Tick' else False

        except Exception as e:
            print(f"Error in audio analysis: {e}")
            return False

    def _detect_snap(self, audio_data):
        """Enhanced snap detection using multiple features"""
        # Initial energy-based detection
        audio_data = librosa.util.normalize(audio_data)
        # sd.play(audio_data, self.RATE)
        # sd.wait()
        # Find peaks in the audio signal
        peaks, _ = find_peaks(np.abs(audio_data),
                              height=self.THRESHOLD,
                              distance=int(self.RATE * 0.1))  # Minimum 100ms between snaps

        if len(peaks) > 0:
            for peak_idx in peaks:
                # Analyze window around peak
                start_idx = max(0, peak_idx - int(self.RATE * 0.05))
                end_idx = min(len(audio_data), peak_idx + int(self.RATE * 0.05))
                peak_segment = audio_data[start_idx:end_idx]
                # sd.play(peak_segment, self.RATE)
                # sd.wait()
                # Perform detailed analysis on the segment
                if self._analyze_audio_features(peak_segment):
                    return True

        return False

    def _audio_process_thread(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK,
                        stream_callback=self._audio_callback)

        stream.start_stream()

        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1)

                print(audio_data)
                if self._detect_snap(audio_data):
                    self.callback()
                    time.sleep(0.5)  # Prevent multiple triggers

            except queue.Empty:
                continue
        stream.stop_stream()
        stream.close()
        p.terminate()