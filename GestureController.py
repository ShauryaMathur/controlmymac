import cv2
import mediapipe as mp
import os
from math import hypot
import threading
import time
from SnapController import AdvancedSnapDetector


class HandGestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.volume_level = 50  # Initial volume level
        self.pinch_distance_state = 0
        self.is_active = False
        # Initialize advanced snap detector
        self.snap_detector = AdvancedSnapDetector(callback=self.toggle_active)

    def toggle_active(self):
        """Toggle gesture recognition state"""
        self.is_active = not self.is_active
        status = "activated" if self.is_active else "deactivated"
        print(f"Gesture recognition {status}")

    def turn_toggle_off(self):
        self.is_active = False
        
    def calculate_distance(self, p1, p2):
        """Calculate distance between two landmarks"""
        print(p1.x,p2.x,p1.y,p2.y)
        print(hypot(p1.x - p2.x, p1.y - p2.y)*142)
        return hypot(p1.x - p2.x, p1.y - p2.y)*142
    
    def execute_command(self, command):
        """Execute MacOS system commands"""
        if command == "volume_up":
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'")
        elif command == "volume_down":
            os.system("osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'")
        elif command == "mute":
            os.system("osascript -e 'set volume with output muted'")
        # elif command == "sleep":
        #     os.system("pmset sleepnow")
    
    def detect_gesture(self, hand_landmarks):
        """Detect specific gestures based on hand landmarks"""
        
        # Count extended fingers for different commands
        fingers_up = self.count_fingers(hand_landmarks)

        if fingers_up == 0:  # Fist (all fingers down)
            return "mute"

        # Thumb and index finger points for volume control
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance for pinch gesture
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        print('Check',pinch_distance,self.pinch_distance_state)

        # Gesture recognition logic
        if abs(pinch_distance - self.pinch_distance_state) > 10:  # Pinch gesture
            
            if pinch_distance > self.pinch_distance_state:
                self.pinch_distance_state = pinch_distance
                return "volume_up"
            else:
                self.pinch_distance_state = pinch_distance
                return "volume_down"
        
        # elif fingers_up == 5:  # All fingers up
        #     return "sleep"
        return None
    
    def count_fingers(self, hand_landmarks):
        """Count number of extended fingers"""
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        finger_count = 0
        
        # Thumb
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x > \
           hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x:
            finger_count += 1
            
        # Other fingers
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < \
               hand_landmarks.landmark[tip - 2].y:
                finger_count += 1
                
        return finger_count

    def run(self):
        """Main application loop"""
        print("Initializing snap detection...")
        print("Please wait while the audio system calibrates...")
        time.sleep(2)  # Give audio system time to initialize

        self.snap_detector.start()
        print("Snap detection active! Snap your fingers to toggle gesture recognition.")

        cap = cv2.VideoCapture(0)

        try:
            while True:
                success, image = cap.read()
                if not success:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                # Display status and audio monitoring indicator
                status_color = (0, 255, 0) if self.is_active else (0, 0, 255)
                status_text = "ACTIVE" if self.is_active else "INACTIVE (Snap to activate)"
                cv2.putText(image, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                # Add audio monitoring indicator
                cv2.circle(image, (20, 60), 5, (0, 255, 0), -1)
                cv2.putText(image, "Listening for snaps", (30, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )

                        if self.is_active:
                            threading.Timer(5, self.turn_toggle_off).start()
                            gesture = self.detect_gesture(hand_landmarks)
                            if gesture:
                                self.execute_command(gesture)

                cv2.imshow("Advanced Snap-Activated Gesture Control", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.snap_detector.stop()
            cap.release()
            cv2.destroyAllWindows()

