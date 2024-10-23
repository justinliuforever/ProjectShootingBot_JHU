## Shooting bot

This project uses YOLOv11 for object detection in images and videos.

## Setup

1. Clone the repository:

   ```
   git clone ....
   cd ...
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the main application:

`python src/main.py`

This will process the image and/or video specified in the config file.

- Press 'q' to quit image detection and move to video detection (if enabled)
- Press 'q' to quit video detection



#### Configuration

Adjust the settings in `src/config.py` to we needs. You can set:

- Model path
- Image and video paths for processing
- Confidence and IOU thresholds
- Whether to process images, videos, or both

#### project Structure

- `src/`: Contains the main source code
  - `main.py`: Main script to run the application
  - `detect_targets.py`: Contains the TargetDetector class and related functions
  - `config.py`: Configuration settings
- `models/`: Store mode (currently we use YOLOv11s model here)
- `data/`: Contains input images and videos for testing

#### Requirements

See `requirements.txt` for a list of required packages.

#### Training tool

Data Resource: roboflow (https://universe.roboflow.com/roboflow-100/csgo-videogame)
Trainning Model: ultralytics+Colab

#### Dependency reference

python3 -m venv venv
Active venv: source venv/bin/activate

(Pip install reference)
pip install torch torchvision torchaudio
pip install opencv-python opencv-python-headless
pip install requests # For dataset fetching
pip install matplotlib # For visualizations if needed
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt

