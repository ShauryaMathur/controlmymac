# ControlMyMac 🖥️🎵👋

**ControlMyMac** is a Python application that activates on the **snap sound** (using YAMNet and Librosa) and reads **hand gestures** (using OpenCV and Mediapipe) to control your Mac's volume and put it to sleep.  

---

## Features 🌟
- **Snap Detection**: Activates the app when a snap sound is detected.
- **Gesture Recognition**: Control your Mac's volume and put it to sleep with intuitive hand gestures.
- **Seamless Integration**: Designed to work efficiently on macOS.

---

## Prerequisites ✅
- **Python 3.7+**
- Recommended: Use a virtual environment to manage dependencies.

---

## Installation & Setup 🛠️

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ShauryaMathur/controlmymac.git
   cd controlmymac
   ```
2. **Set Up a Virtual Environment(Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   .\venv\Scripts\activate   # Windows
   ```
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app
   ```bash
   python controlmymac.py
   ```

Notes 📝
	•	This application uses YAMNet and Librosa for sound detection.
	•	Gesture recognition is powered by OpenCV and Mediapipe.
	•	Ensure your system’s microphone and camera are functioning properly for optimal performance.
