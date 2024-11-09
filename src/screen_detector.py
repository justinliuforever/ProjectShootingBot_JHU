import cv2
import numpy as np
import pyautogui
from detect_targets import TargetDetector
import config
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
import sys
import time
from pynput import keyboard, mouse

class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, window, detector):
        super().__init__()
        self.window = window
        self.detector = detector
        self.is_running = False
        self.is_capturing = True
        self.last_time = time.time()
        self.fps = 0
        
        # Simplified auto-aim settings
        self.should_aim = False  # Flag for single click aim
        
        # Initialize mouse control
        pyautogui.PAUSE = 0.005
        pyautogui.FAILSAFE = False
        
        # Initialize keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            suppress=True  # Prevent key event propagation
        )
        self.keyboard_listener.start()
        
        # Add detection window size
        self.detection_window_size = (640, 416)  # Default size matching your model's input

    def run(self):
        while self.is_capturing:
            try:
                if self.is_running:
                    # Calculate FPS
                    current_time = time.time()
                    self.fps = 1 / (current_time - self.last_time)
                    self.last_time = current_time

                    frame = self.capture_screen()
                    if frame is not None:
                        # Detect targets - now returns box coordinates directly
                        boxes = self.detector.detect(frame)
                        
                        # Handle auto-aim if enabled
                        if self.should_aim and boxes:
                            self.handle_auto_aim(boxes)
                        
                        # Draw interface
                        frame = self.draw_interface(frame, boxes)
                        self.frame_ready.emit(frame)
                    
                time.sleep(0.001)
            except Exception as e:
                print(f"Processing error: {e}")
                continue

    def handle_auto_aim(self, results):
        """Move mouse to detected target position"""
        try:
            if not self.should_aim or not results:
                return

            # Get window position
            window_x = self.window.x()
            window_y = self.window.y() + 30  # MacOS title bar offset

            # Get the first detected target
            target = results[0]  # Format: [x1, y1, x2, y2]
            
            # Calculate target center in detection frame
            target_center_x = (target[0] + target[2]) / 2  # Average of x1 and x2
            target_center_y = (target[1] + target[3]) / 2  # Average of y1 and y2
            
            # Convert to screen coordinates
            screen_x = window_x + target_center_x
            screen_y = window_y + target_center_y
            
            # Add debug prints
            print(f"Window position: ({window_x}, {window_y})")
            print(f"Target center in frame: ({target_center_x}, {target_center_y})")
            print(f"Moving to screen position: ({screen_x}, {screen_y})")
            
            # Move mouse to target
            pyautogui.moveTo(
                int(screen_x),
                int(screen_y),
                duration=0.1,
                tween=pyautogui.easeOutQuad
            )
            
            # Reset flag after moving
            self.should_aim = False

        except Exception as e:
            print(f"Auto-aim error: {str(e)}")
            self.should_aim = False

    def find_best_target(self, results):
        """Find the target closest to center of the frame"""
        if not results:
            return None

        # Get frame center
        center_x = self.detection_window_size[0] / 2
        center_y = self.detection_window_size[1] / 2

        # Find closest target to center
        closest_target = None
        min_distance = float('inf')

        for box in results:
            if len(box) < 4:  # Ensure box has enough coordinates
                continue
                
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2
            
            distance = ((box_center_x - center_x) ** 2 + 
                       (box_center_y - center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_target = box

        return closest_target

    def draw_interface(self, frame, results):
        """Draw detection boxes and interface elements"""
        # Draw original boxes
        frame = self.detector.draw_boxes(frame, results)
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {int(self.fps)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw auto-aim status
        status = "Auto-aim: ON" if self.should_aim else "Auto-aim: OFF"
        cv2.putText(frame, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw target indicator if auto-aim is enabled
        if self.should_aim and results:
            target = self.find_best_target(results)
            if target is not None:
                center_x = int((target[0] + target[2]) / 2)
                center_y = int((target[1] + target[3]) / 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Target: ({center_x}, {center_y})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def on_key_press(self, key):
        """Handle keyboard press events"""
        try:
            if key == keyboard.KeyCode.from_char('f'):
                self.should_aim = True
                print("Auto-aim triggered")
        except AttributeError:
            pass
        return True

    def capture_screen(self):
        try:
            x = self.window.x()
            y = self.window.y()
            width = self.window.width()
            height = self.window.height() - 50  # Subtract button area height
            
            # Compensate for MacOS title bar height
            y += 30
            
            if width <= 0 or height <= 0:
                return None
                
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"Screenshot error: {e}")
            return None

class ResizableTransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # Initialize detector
        self.detector = TargetDetector(config.MODEL_PATH, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
        
        # Create capture thread
        self.capture_thread = CaptureThread(self, self.detector)
        self.capture_thread.frame_ready.connect(self.show_frame)
        self.capture_thread.start()
        
        # Create result window
        cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Results', 400, 300)
        
        # Resize parameters
        self.resizing = False
        self.resize_margin = 40  # Pixels to detect resize area
        self.resize_edge = None
        self.resize_sensitivity = 0.1  # Add resize sensitivity control (1.0 = normal, < 1.0 = slower)

    def initUI(self):
        # Set window properties
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create semi-transparent background frame
        frame = QWidget()
        frame.setStyleSheet('''
            QWidget {
                background-color: rgba(255, 255, 255, 30);
                border: 2px solid red;
                border-radius: 5px;
            }
        ''')
        layout.addWidget(frame)
        
        # Create button layout
        button_layout = QVBoxLayout()
        
        # Create buttons
        self.toggle_btn = QPushButton('Start Detection')
        self.toggle_btn.clicked.connect(self.toggle_detection)
        self.toggle_btn.setStyleSheet('''
            QPushButton {
                background-color: green;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        ''')
        
        quit_btn = QPushButton('Quit')
        quit_btn.clicked.connect(self.close)
        quit_btn.setStyleSheet('''
            QPushButton {
                background-color: red;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
        ''')
        
        button_layout.addWidget(self.toggle_btn)
        button_layout.addWidget(quit_btn)
        layout.addLayout(button_layout)
        
        # Set initial window size and position
        self.setGeometry(100, 100, 400, 300)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if clicking in resize area
            edge = self.get_resize_edge(event.pos())
            if edge:
                self.resizing = True
                self.resize_edge = edge
                self.resize_start_pos = event.globalPos()
                self.resize_start_geometry = self.geometry()
            else:
                # Normal window dragging
                self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self.resize_edge = None

    def mouseMoveEvent(self, event):
        if self.resizing and self.resize_edge:
            # Handle resizing with sensitivity adjustment
            delta = event.globalPos() - self.resize_start_pos
            new_geometry = self.resize_start_geometry
            
            # Apply sensitivity to delta
            delta_x = int(delta.x() * self.resize_sensitivity)
            delta_y = int(delta.y() * self.resize_sensitivity)
            
            if 'right' in self.resize_edge:
                new_width = max(100, self.resize_start_geometry.width() + delta_x)
                # Ensure smooth movement by limiting change per frame
                current_width = self.width()
                max_change = 2  # Maximum pixels to change per frame
                new_width = current_width + max(min(new_width - current_width, max_change), -max_change)
                new_geometry.setWidth(new_width)
                
            if 'bottom' in self.resize_edge:
                new_height = max(100, self.resize_start_geometry.height() + delta_y)
                current_height = self.height()
                max_change = 2
                new_height = current_height + max(min(new_height - current_height, max_change), -max_change)
                new_geometry.setHeight(new_height)
                
            if 'left' in self.resize_edge:
                new_width = max(100, self.resize_start_geometry.width() - delta_x)
                current_width = self.width()
                max_change = 2
                new_width = current_width + max(min(new_width - current_width, max_change), -max_change)
                new_x = self.resize_start_geometry.right() - new_width
                new_geometry.setLeft(new_x)
                
            if 'top' in self.resize_edge:
                new_height = max(100, self.resize_start_geometry.height() - delta_y)
                current_height = self.height()
                max_change = 2
                new_height = current_height + max(min(new_height - current_height, max_change), -max_change)
                new_y = self.resize_start_geometry.bottom() - new_height
                new_geometry.setTop(new_y)
            
            self.setGeometry(new_geometry)
            
        elif hasattr(self, 'oldPos'):
            # Handle window dragging with smoother movement
            delta = event.globalPos() - self.oldPos
            max_move = 20  # Maximum pixels to move per frame
            delta_x = max(min(delta.x(), max_move), -max_move)
            delta_y = max(min(delta.y(), max_move), -max_move)
            self.move(self.x() + delta_x, self.y() + delta_y)
            self.oldPos = event.globalPos()
        
        # Update cursor and border color
        edge = self.get_resize_edge(event.pos())
        self._update_cursor(edge)
        self._update_border_color(edge)

    def _update_cursor(self, edge):
        """Helper method to update cursor based on edge"""
        if edge in ['left', 'right']:
            self.setCursor(Qt.SizeHorCursor)
        elif edge in ['top', 'bottom']:
            self.setCursor(Qt.SizeVerCursor)
        elif edge in ['topleft', 'bottomright']:
            self.setCursor(Qt.SizeFDiagCursor)
        elif edge in ['topright', 'bottomleft']:
            self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _update_border_color(self, edge):
        """Helper method to update border color"""
        color = 'yellow' if edge else 'red'
        self.centralWidget().setStyleSheet(f'''
            QWidget {{
                background-color: rgba(255, 255, 255, 30);
                border: 2px solid {color};
                border-radius: 5px;
            }}
        ''')

    def get_resize_edge(self, pos):
        # Determine which edge (if any) the position is near
        margin = self.resize_margin
        width = self.width()
        height = self.height()
        
        if pos.x() < margin:
            if pos.y() < margin:
                return 'topleft'
            elif pos.y() > height - margin:
                return 'bottomleft'
            return 'left'
        elif pos.x() > width - margin:
            if pos.y() < margin:
                return 'topright'
            elif pos.y() > height - margin:
                return 'bottomright'
            return 'right'
        elif pos.y() < margin:
            return 'top'
        elif pos.y() > height - margin:
            return 'bottom'
        return None

    def toggle_detection(self):
        self.capture_thread.is_running = not self.capture_thread.is_running
        btn_text = "Stop Detection" if self.capture_thread.is_running else "Start Detection"
        btn_color = 'red' if self.capture_thread.is_running else 'green'
        self.toggle_btn.setText(btn_text)
        self.toggle_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {btn_color};
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }}
        ''')

    def show_frame(self, frame):
        cv2.imshow('Detection Results', frame)
        cv2.waitKey(1)

    def closeEvent(self, event):
        self.capture_thread.is_capturing = False
        self.capture_thread.wait()
        cv2.destroyAllWindows()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ResizableTransparentWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
