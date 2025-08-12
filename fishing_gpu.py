import sys
import numpy as np
import mss
import cv2
from ultralytics import YOLO
import time
import os
import pydirectinput  # Use this instead of pyautogui for games
import random
import time
import torch  # Add torch import for GPU detection
import tkinter as tk
from tkinter import ttk
import threading

# Model path for YOLO (relative to script location)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# Condition initialization image path (relative to script location)
COND_INIT_PATH = os.path.join(os.path.dirname(__file__), "cond_init.png")

# Specific region for HSV gradient detection
HSV_MONITOR = {"top": 230, "left": 745, "width": 427, "height": 25}

# Entire screen for YOLO fish detection - adjust as needed for your screen
FULL_SCREEN = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust to your screen resolution

# HSV ranges for gradients
GRADIENTS = [
    {"name": "green",  "lower": (40, 40, 80),  "upper": (70, 160, 255), "color": (0, 255, 0)},
    {"name": "purple", "lower": (130, 40, 80), "upper": (170, 255, 255), "color": (128, 0, 128)},
    {"name": "gray",   "lower": (0, 0, 60),    "upper": (180, 70, 180), "color": (128, 128, 128)},
    {"name": "yellow", "lower": (20, 60, 90),  "upper": (35, 255, 255), "color": (0, 255, 255)},
]

# Detection mode: "hsv", "yolo", or "both"
MODE = "both"  # Default to both

# Bot configuration
BOT_ENABLED = False  # Start with bot disabled
LAST_ACTION_TIME = 0
ACTION_COOLDOWN = 1.0  # Seconds between actions
INIT_MODE = True  # Start in initialization mode (looking for cond_init)
FISHING_START_TIME = 0  # Track when fishing mode started
FISHING_TIMEOUT = 10.0  # Seconds to fish before resetting to init mode

# Device selection - will be set to CUDA for GPU acceleration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def find_template(frame, template, threshold=0.8):
    """Find template in frame using template matching"""
    if template is None:
        return None
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame
    
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # Perform template matching
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        h, w = template_gray.shape
        return {
            "x": max_loc[0],
            "y": max_loc[1],
            "w": w,
            "h": h,
            "confidence": max_val
        }
    return None

def detect_hsv(frame):
    """Detect gradients using HSV color ranges"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # We'll keep track of the best box (largest area)
    best_box = None
    max_area = 0
    
    for grad in GRADIENTS:
        mask = cv2.inRange(hsv, np.array(grad["lower"]), np.array(grad["upper"]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_area:  # Keep only the largest area
                max_area = area
                x, y, w, h = cv2.boundingRect(cnt)
                best_box = {
                    "name": grad["name"],
                    "color": grad["color"],
                    "x": x, 
                    "y": y, 
                    "w": w, 
                    "h": h,
                    "area": area,
                    "type": "gradient"
                }
    
    return best_box

def detect_yolo(frame, model):
    """Detect objects using YOLO model"""
    # Ensure the frame is in the right format for YOLO
    if frame.shape[2] == 4:  # Check if we have an alpha channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Run inference with GPU acceleration
    results = model(frame, conf=0.3, device=DEVICE)
    
    # Find the best detection
    best_box = None
    max_conf = 0
    
    if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        for i in range(len(results[0].boxes)):
            box = results[0].boxes.xyxy[i].cpu().numpy()
            conf = results[0].boxes.conf[i].item()
            
            # Get class name if available
            class_id = 0
            if hasattr(results[0].boxes, 'cls') and len(results[0].boxes.cls) > i:
                class_id = int(results[0].boxes.cls[i].item())
            class_name = model.names[class_id] if hasattr(model, 'names') else "Fish"
            
            if conf > max_conf:
                max_conf = conf
                x1, y1, x2, y2 = map(int, box)
                best_box = {
                    "name": f"{class_name} {conf:.2f}",
                    "color": (0, 0, 255),  # Red for YOLO detections
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "conf": conf,
                    "type": "fish"
                }
    
    return best_box

def check_overlap(box1, box2):
    """Check if two boxes overlap"""
    if not box1 or not box2:
        return False
    
    # For box1, adjust to screen coordinates if it's a gradient box
    x1 = box1.get("x_screen", box1["x"])
    y1 = box1.get("y_screen", box1["y"])
    w1 = box1["w"]
    h1 = box1["h"]
    
    # For box2, adjust to screen coordinates if it's a gradient box
    x2 = box2.get("x_screen", box2["x"])
    y2 = box2.get("y_screen", box2["y"])
    w2 = box2["w"]
    h2 = box2["h"]
    
    # Check for overlap
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def perform_action():
    """Press spacebar when fish and gradient overlap or when initializing"""
    global LAST_ACTION_TIME
    
    # Check cooldown
    current_time = time.time()
    if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
        return
    
    # Press spacebar
    pydirectinput.press('space')
    LAST_ACTION_TIME = current_time
    
    # Small random delay to seem more human-like
    time.sleep(0.05 + random.random() * 0.1)

class FishingBotGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fishing Bot Control")
        self.root.geometry("400x320")
        self.root.resizable(False, False)
        
        # Bot control variables
        self.bot_running = False
        self.detection_thread = None
        self.stop_detection = threading.Event()
        
        # Initialize bot components
        self.model = None
        self.cond_init_template = None
        
        # Load YOLO model
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded successfully: {MODEL_PATH}")
            
            # Test model inference with GPU
            test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            test_results = self.model(test_frame, conf=0.3, device=DEVICE)
            print(f"✅ Model inference test successful on {DEVICE.upper()}")
            
        except Exception as e:
            print(f"❌ Failed to load or test model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
        
        # Load condition initialization template
        if os.path.exists(COND_INIT_PATH):
            self.cond_init_template = cv2.imread(COND_INIT_PATH)
            print(f"✅ Template loaded: {COND_INIT_PATH}")
        else:
            print(f"⚠️ Template not found: {COND_INIT_PATH}")
            self.cond_init_template = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Fishing Bot Controller", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=10)
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Bot: DISABLED", 
                                   font=("Arial", 12), fg="red")
        self.status_label.pack()
        
        self.mode_label = tk.Label(status_frame, text="Mode: STANDBY", 
                                 font=("Arial", 10), fg="gray")
        self.mode_label.pack()
        
        self.device_label = tk.Label(status_frame, text=f"Device: {DEVICE.upper()}", 
                                   font=("Arial", 10), fg="blue")
        self.device_label.pack()
        
        # Add detection status
        self.detection_label = tk.Label(status_frame, text="Detections: None", 
                                      font=("Arial", 9), fg="gray")
        self.detection_label.pack()
        
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=20, pady=10)
        
        self.start_button = tk.Button(control_frame, text="Start Bot", 
                                    command=self.toggle_bot, 
                                    font=("Arial", 12, "bold"),
                                    bg="green", fg="white", width=15)
        self.start_button.pack(pady=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self.root, text="Statistics", padding=10)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        self.actions_label = tk.Label(stats_frame, text="Actions performed: 0", 
                                    font=("Arial", 10))
        self.actions_label.pack()
        
        self.runtime_label = tk.Label(stats_frame, text="Runtime: 0s", 
                                    font=("Arial", 10))
        self.runtime_label.pack()
        
        # Footer
        footer_label = tk.Label(self.root, text="Press 'Start Bot' to begin fishing automation", 
                              font=("Arial", 9), fg="gray")
        footer_label.pack(side="bottom", pady=10)
        
        # Initialize stats
        self.actions_count = 0
        self.start_time = None
        self.detection_status = "Detections: None"
        
        # Start UI update loop
        self.update_ui()
        
    def toggle_bot(self):
        if not self.bot_running:
            self.start_bot()
        else:
            self.stop_bot()
    
    def start_bot(self):
        self.bot_running = True
        self.start_time = time.time()
        self.actions_count = 0
        self.stop_detection.clear()
        
        # Update UI
        self.start_button.config(text="Stop Bot", bg="red")
        self.status_label.config(text="Bot: ENABLED", fg="green")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
    
    def stop_bot(self):
        self.bot_running = False
        self.stop_detection.set()
        
        # Update UI
        self.start_button.config(text="Start Bot", bg="green")
        self.status_label.config(text="Bot: DISABLED", fg="red")
        self.mode_label.config(text="Mode: STANDBY")
    
    def detection_loop(self):
        global BOT_ENABLED, INIT_MODE, FISHING_START_TIME, LAST_ACTION_TIME
        
        # Create MSS instance for this thread
        sct = mss.mss()
        
        BOT_ENABLED = True
        INIT_MODE = True
        loop_count = 0
        
        print(f"🔄 Detection loop started with device: {DEVICE}")
        
        while self.bot_running and not self.stop_detection.is_set():
            try:
                loop_count += 1
                
                if INIT_MODE:
                    # Look for initialization condition
                    yolo_img = np.array(sct.grab(FULL_SCREEN))
                    yolo_frame = cv2.cvtColor(yolo_img, cv2.COLOR_BGRA2BGR)
                    
                    if self.cond_init_template is not None:
                        init_match = find_template(yolo_frame, self.cond_init_template, 0.8)
                        if init_match:
                            print("✅ Init condition found, switching to fishing mode")
                            INIT_MODE = False
                            FISHING_START_TIME = time.time()
                        else:
                            if loop_count % 100 == 0:  # Log every 100 iterations
                                print(f"🔍 Looking for init condition... (loop {loop_count})")
                            # No init condition found, press space to initialize
                            self.perform_action()
                    else:
                        # No template available, skip init mode
                        print("⚠️ No template available, skipping to fishing mode")
                        INIT_MODE = False
                        FISHING_START_TIME = time.time()
                else:
                    # Fishing mode: check for overlap between YOLO and HSV
                    current_time = time.time()
                    
                    # Check if fishing timeout reached - reset to init mode
                    if current_time - FISHING_START_TIME > FISHING_TIMEOUT:
                        print("⏰ Fishing timeout reached, resetting to init mode")
                        INIT_MODE = True
                        continue
                    
                    # Capture screens
                    yolo_img = np.array(sct.grab(FULL_SCREEN))
                    yolo_frame = cv2.cvtColor(yolo_img, cv2.COLOR_BGRA2BGR)
                    
                    hsv_img = np.array(sct.grab(HSV_MONITOR))
                    hsv_frame = cv2.cvtColor(hsv_img, cv2.COLOR_BGRA2BGR)
                    
                    gradient_box = None
                    fish_box = None
                    
                    # HSV detection
                    if MODE == "hsv" or MODE == "both":
                        gradient_box = detect_hsv(hsv_frame)
                        if gradient_box:
                            # Adjust coordinates to full screen reference
                            gradient_box["x_screen"] = gradient_box["x"] + HSV_MONITOR["left"]
                            gradient_box["y_screen"] = gradient_box["y"] + HSV_MONITOR["top"]
                    
                    # YOLO detection
                    if (MODE == "yolo" or MODE == "both") and self.model:
                        fish_box = detect_yolo(yolo_frame, self.model)
                    
                    # Debug logging every 100 iterations
                    if loop_count % 100 == 0:
                        status = f"🎣 Fishing... Gradient: {'✅' if gradient_box else '❌'} | Fish: {'✅' if fish_box else '❌'}"
                        print(status)
                        # Update UI with detection status
                        self.detection_status = status
                    
                    # Check for overlap and perform action
                    if gradient_box and fish_box:
                        if check_overlap(gradient_box, fish_box):
                            print("🎉 OVERLAP DETECTED! Performing action...")
                            time.sleep(0.12)  # Small delay before action
                            self.perform_action()
                            # After catching fish, reset to init mode after a short delay
                            time.sleep(2.0)
                            INIT_MODE = True
                            continue
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Detection loop error: {e}")
                continue
        
        print("🛑 Detection loop stopped")
    
    def perform_action(self):
        """Press spacebar when fish and gradient overlap or when initializing"""
        global LAST_ACTION_TIME
        
        # Check cooldown
        current_time = time.time()
        if current_time - LAST_ACTION_TIME < ACTION_COOLDOWN:
            return
        
        # Press spacebar
        pydirectinput.press('space')
        LAST_ACTION_TIME = current_time
        self.actions_count += 1
        
        # Small random delay to seem more human-like
        time.sleep(0.05 + random.random() * 0.1)
    
    def update_ui(self):
        """Update UI elements periodically"""
        if self.bot_running:
            # Update mode
            mode_text = "INIT" if INIT_MODE else f"FISHING ({FISHING_TIMEOUT - (time.time() - FISHING_START_TIME):.1f}s)"
            self.mode_label.config(text=f"Mode: {mode_text}")
            
            # Update runtime
            if self.start_time:
                runtime = int(time.time() - self.start_time)
                self.runtime_label.config(text=f"Runtime: {runtime}s")
            
            # Update actions count
            self.actions_label.config(text=f"Actions performed: {self.actions_count}")
            
            # Update detection status
            self.detection_label.config(text=self.detection_status)
        
        # Schedule next update
        self.root.after(100, self.update_ui)
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        if self.bot_running:
            self.stop_bot()
        self.root.destroy()

def main():
    try:
        print("🤖 Starting Fishing Bot GUI...")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🎯 Model path: {MODEL_PATH}")
        print(f"🖼️ Template path: {COND_INIT_PATH}")
        print(f"🚀 Device: {DEVICE.upper()} {'(GPU Acceleration)' if DEVICE == 'cuda' else '(CPU)'}")
        
        app = FishingBotGUI()
        print("🎮 GUI window should now be visible")
        app.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
