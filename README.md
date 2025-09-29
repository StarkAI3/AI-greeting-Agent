# Lost & Found - Advanced Face Recognition System

## Overview
Lost & Found is a state-of-the-art face recognition system designed for real-time identification, video analysis, and database management. Built with modern AI and web technologies, it provides a seamless workflow for uploading, processing, and recognizing faces from images, videos, and live camera feeds.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Recommended: Python 3.10)
- **CUDA-compatible GPU** (Optional but recommended for better performance)
- **Pinecone API Key** (Get from [pinecone.io](https://pinecone.io))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OmkarDeshpande777/Lost-Found-Face-Recognition.git
   cd "Lost & Found with new Face pt model"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

4. **Ensure YOLO model files are in place:**
   - Face detection model: `Models/Face_Detectbest.pt`

### Running the Application

```bash
python app.py
```
- **URL:** http://localhost:5000
- **Features:** Face detection, recognition, enrollment, video processing, live camera feed

---

## 🛠️ Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: PyTorch, FaceNet (facenet-pytorch), YOLO8n
- **Database**: Pinecone (vector database for embeddings)
- **Image/Video Processing**: OpenCV
- **GPU Acceleration**: NVIDIA CUDA

---

## 📁 Project Structure

```
Lost & Found with new Face pt model/
├── 📄 app.py                           # Main face recognition Flask application
├── 📄 model3.py                        # Face recognition model (YOLO8n + FaceNet)
├── 🌐 face_recognition_advanced.html   # Advanced web interface
├── 🌐 face_recognition.html           # Basic web interface
├── 📋 requirements.txt                 # Python dependencies
├── 🔧 .env                            # Environment variables (API keys)
├── 📊 README.md                        # Project documentation
├── 📖 DEVELOPER_REFERENCE.md          # Technical documentation
├── 🚫 .gitignore                      # Git ignore patterns
├── 📁 Models/                          # AI model files
│   └── Face_Detectbest.pt            # YOLO face detection model
├── 📁 enrolled_faces/                  # Stored face images and metadata
├── 📁 uploads/                        # User uploaded files
├── 📁 outputs/                        # Processed output files
└── 📁 static/                         # Static web assets
```

## 🎯 Core Files Explained

### **Application**
- **`app.py`** - Main Flask app for face recognition
  - Handles face detection, recognition, enrollment
  - Serves web interface at `/`
  - Processes images/videos for face identification
  - Provides real-time camera feed processing

### **AI Model**
- **`model3.py`** - Face Recognition Model
  - Uses YOLO8n for face detection
  - Uses FaceNet for face embedding extraction
  - Manages Pinecone vector database operations
  - Handles face enrollment and recognition

### **Web Interface**
- **`face_recognition_advanced.html`** - Feature-rich web UI
  - Camera controls, batch processing, analytics
  - Advanced settings and real-time recognition

- **`face_recognition.html`** - Simple web interface
  - Basic upload and recognition functionality

---

## 🧩 Key Functions & Endpoints

### Face Recognition App (`app.py`)
- `/` - Main web interface
- `/api/health` - System health & GPU status
- `/api/enroll` - Enroll new faces
- `/api/process` - Process uploaded media
- `/api/faces` - Manage enrolled faces
- `/api/camera/start` - Start live camera feed
- `/api/analytics` - View system analytics

---

## ✨ Features
- **Real-time Camera Recognition** - Live face detection from webcam
- **Video Processing** - Batch process videos with face detection
- **Face Database Management** - Add, update, delete enrolled faces
- **GPU Acceleration** - Automatic CUDA support for better performance
- **Web Interface** - User-friendly HTML interface
- **RESTful API** - Complete API for integration
- **Analytics Dashboard** - Performance metrics and statistics
---

## 🚀 Usage Guide

### Face Recognition Workflow
1. **Start the application:** `python app.py`
2. **Open browser:** Navigate to `http://localhost:5000`
3. **Enroll faces:** Upload images to build your face database
4. **Process media:** Upload images/videos for face recognition
5. **Live recognition:** Use camera feed for real-time identification

---

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```env
PINECONE_API_KEY=your_pinecone_api_key_here
```

### Model Files Required
Ensure this file exists:
- `Models/Face_Detectbest.pt` - YOLO face detection model

### GPU Setup (Optional)
For CUDA GPU acceleration:
1. Install CUDA toolkit
2. Install PyTorch with CUDA support
3. System will automatically detect and use GPU

---

## 📊 Performance
- **GPU Accelerated:** RTX 4060 or better recommended
- **Memory:** 8GB+ RAM, 6GB+ VRAM for optimal performance
- **Storage:** SSD recommended for faster model loading
- **Processing Speed:** ~30-50 FPS on modern GPUs

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## � License
This project is open source. Please ensure compliance with model licensing terms.

---

## 🆘 Troubleshooting

### Common Issues
- **ModuleNotFoundError:** Run `pip install -r requirements.txt`
- **CUDA not found:** Install NVIDIA CUDA toolkit
- **Pinecone errors:** Check API key in `.env` file
- **Model not found:** Verify YOLO model files are in `Models/` directory

### Support
For issues and questions, please open an issue on GitHub.

---

## � Acknowledgments
- **YOLO:** Ultralytics for object detection
- **FaceNet:** For face recognition embeddings  
- **Pinecone:** For vector database services
- **PyTorch:** For deep learning framework

---

## 🏆 Credits
Developed by Omkar Deshpande and contributors.
