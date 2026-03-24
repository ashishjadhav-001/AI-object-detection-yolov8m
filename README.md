# 🔍 AI Object Detection System (YOLOv8m)

## 🚀 Overview

The **AI Object Detection System** is a real-time computer vision application that detects objects from images and webcam input using the powerful **YOLOv8m model**.

This project demonstrates the ability to build an end-to-end AI application combining **Deep Learning, Computer Vision, and UI development**.

---

## 🎯 Key Features

* 📤 Upload image for object detection
* 📸 Capture image using webcam
* 🔍 Detect multiple objects with bounding boxes
* 🏷️ Display object labels clearly
* 🔢 Count number of detected objects
* ⚙️ Adjustable confidence threshold
* 🎨 Clean and interactive Streamlit UI

---

## 🧠 How It Works

1. **Input Handling**

   * Accepts image upload or webcam capture via Streamlit

2. **Preprocessing**

   * Converts image from RGB → BGR format for model compatibility

3. **Model Inference**

   * Uses **YOLOv8m (Ultralytics)** pretrained on COCO dataset
   * Detects objects and returns bounding boxes with confidence scores

4. **Postprocessing**

   * Extracts labels and counts objects
   * Converts output back to RGB for correct display

5. **Visualization**

   * Displays annotated image with bounding boxes
   * Shows detected object count and labels

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning Model:** YOLOv8m (Ultralytics)
* **Computer Vision:** OpenCV
* **Frontend/UI:** Streamlit
* **Libraries:** NumPy, Pillow

---

## 📁 Project Structure

```text
AI-Object-Detection-YOLOv8/
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/ashishjadhav-001/AI-object-detection-yolov8m
cd AI-object-detection-yolov8m
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Example Output

* Bounding boxes drawn around detected objects
* Labels such as **person, car, laptop, bottle**
* Total object count displayed
* Clean dashboard-style UI

---

## 🔥 Key Highlights

* Built a **real-time object detection system** using YOLOv8m
* Integrated both **image upload and webcam input**
* Implemented **object counting and label extraction**
* Designed a **dashboard-style UI with Streamlit**
* Handled **RGB ↔ BGR conversion** for correct visualization

---

## 🚀 Future Improvements

* Add video stream detection
* Custom object detection (e.g., pen detection)
* Model optimization for faster inference
* Deploy using cloud platforms

---

## 💼 Use Cases

* Smart surveillance systems
* Traffic monitoring
* Retail analytics (people counting)
* Security applications

---

## 👨‍💻 Author

**Ashish Jadhav (AJ)**

---

## ⭐ Conclusion

This project demonstrates the ability to build and deploy a real-world **AI-powered computer vision system**, combining deep learning, real-time inference, and user-friendly interface design.
