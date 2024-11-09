import pyautogui
from pynput import keyboard
import sys
from PyQt5.QtWidgets import QApplication
import time

# Disable PyAutoGUI's fail-safe
pyautogui.FAILSAFE = False

class MouseController:
    def __init__(self):
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Add small pause to make movements more reliable
        pyautogui.PAUSE = 0.005
        
        # Initialize keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        print("Press 'T' to move mouse to screen center")
        print("Press 'ESC' to exit")
        print("\nIMPORTANT: On macOS, you need to grant accessibility permissions:")
        print("1. Go to System Preferences > Security & Privacy > Accessibility")
        print("2. Allow applications(Virtual Studio Code, Terminal, etc.) to control your computer")

    def on_key_press(self, key):
        try:
            # Test mouse movement with 'T' key
            if hasattr(key, 'char') and key.char == 't':
                self.move_to_center()
            # Exit with ESC key
            elif key == keyboard.Key.esc:
                print("Exiting...")
                self.keyboard_listener.stop()
                sys.exit()
        except AttributeError:
            pass

    def move_to_center(self):
        try:
            # Calculate screen center
            center_x = self.screen_width // 2
            center_y = self.screen_height // 2
            
            print(f"Moving mouse to center: ({center_x}, {center_y})")
            
            # Use duration parameter for smoother movement
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            print("Mouse moved using PyAutoGUI")

        except Exception as e:
            print(f"Error moving mouse: {e}")

def main():
    app = QApplication(sys.argv)
    controller = MouseController()
    
    # Keep the program running
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
