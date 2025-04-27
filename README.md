# Face Detection and Recognition Mock Project

Welcome!  
This project demonstrates a simple **face recognition system** that can recognize celebrity faces in real-time using your webcam.  
It uses a **free Kaggle dataset** to train on celebrities' faces and allows easy validation, testing, and accuracy evaluation.

---

## 🚀 Setup and Usage Instructions

1. Clone this repository
```
git clone https://github.com/ruchirk22/Face-Detection-Mock.git
cd Face-Detection-Mock
```

2. Create and activate virtual environment
```
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate   # Mac/Linux
```

3. Install required packages
```
pip install -r requirements.txt
```

4. Download Dataset
Go to datasets/ folder
```
cd datasets
```
(Follow instructions inside datasets/ to download from Kaggle)
After downloading, come back to project root
```
cd ..
```

5. Prepare Validation Folder
```
# Create a folder named 'validation' at the project root
# Structure should be like:
# validation/
# ├── Celebrity1/
# │    ├── image1.jpg
# │    ├── image2.jpg
# ├── Celebrity2/
# │    ├── image1.jpg
# │    ├── image2.jpg
# (Folder names must match dataset names exactly)
```

6. Encode faces and generate encodings.pkl
```
python encode_faces.py
```

7. Run Real-Time Face Recognition using webcam
```
python recognize_faces.py
```

8. Evaluate model accuracy using validation images
```
python eval_accuracy.py
```
