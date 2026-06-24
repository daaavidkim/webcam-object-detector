# AI Security Camera

A real-time security camera application built with Python, OpenCV, and YOLO. 
The system detects security-relevant objects, records events, captures evidence images, records video clips, 
and integrates a custom-trained firearm detection model.

## Features

- Real-time object detection
- Detects people, cars, trucks, and motorcycles
- Track Enter/Exits
- Event logging to CSV
- Automatic image capture Enter/Exit
- Video recording during object detection
- Custom firearm detection model
- Configurable settings through `config.json`

## Technologies

- Python
- OpenCV
- YOLO11
- PyTorch

## Structure

- `webcam_detector.py`    Main security camera application
- `test_tracking.py`      ID tracking experiment
- `test_firearm.py`       Firearm model testing
- `training/`             Firearm model training files
- `config.json`           Application settings

## How to Run

### Clone the Repository

```bash
git clone https://github.com/daaavidkim/webcam-object-detector.git
cd webcam-object-detector
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

**Windows PowerShell**

```powershell
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install ultralytics opencv-python torch
```

### Run the Application

```bash
python webcam_detector.py
```
