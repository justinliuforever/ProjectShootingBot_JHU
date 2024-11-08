import cv2
import numpy as np
import pyautogui
from detect_targets import TargetDetector
import config
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import time

class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, window, detector):
        super().__init__()
        self.window = window
        self.detector = detector
        self.is_running = False
        self.is_capturing = True

    def run(self):
        while self.is_capturing:
            try:
                if self.is_running:
                    frame = self.capture_screen()
                    if frame is not None:
                        results = self.detector.detect(frame)
                        frame = self.detector.draw_boxes(frame, results)
                        self.frame_ready.emit(frame)
                time.sleep(0.01)
            except Exception as e:
                print(f"处理错误: {e}")
                continue

    def capture_screen(self):
        try:
            x = self.window.x()
            y = self.window.y()
            width = self.window.width()
            height = self.window.height() - 50  # 减去按钮区域高度
            
            # 补偿 MacOS 的标题栏高度
            y += 30
            
            if width <= 0 or height <= 0:
                return None
                
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"截图错误: {e}")
            return None

class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # 初始化检测器
        self.detector = TargetDetector(config.MODEL_PATH, config.CONF_THRESHOLD, config.IOU_THRESHOLD)
        
        # 创建捕获线程
        self.capture_thread = CaptureThread(self, self.detector)
        self.capture_thread.frame_ready.connect(self.show_frame)
        self.capture_thread.start()
        
        # 创建结果显示窗口
        cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection Results', 400, 300)

    def initUI(self):
        # 设置窗口属性
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建中心部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 创建一个半透明的背景框
        frame = QWidget()
        frame.setStyleSheet('''
            QWidget {
                background-color: rgba(255, 255, 255, 30);
                border: 2px solid red;
                border-radius: 5px;
            }
        ''')
        layout.addWidget(frame)
        
        # 创建按钮布局
        button_layout = QVBoxLayout()
        
        # 创建按钮
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
        
        # 设置窗口大小和位置
        self.setGeometry(100, 100, 400, 300)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'oldPos'):
            delta = event.globalPos() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

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
    window = TransparentWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
