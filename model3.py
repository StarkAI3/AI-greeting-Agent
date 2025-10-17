import cv2
import torch
import numpy as np
import os
import logging
import json
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import tempfile
import shutil

# Configure torch serialization for YOLO model compatibility
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from facenet_pytorch import InceptionResnetV1
    from ultralytics import YOLO
    from pinecone import Pinecone, ServerlessSpec
    import torchvision.transforms as transforms
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    logging.error(f"Missing required package: {e}")
    print(f"Please install missing package: {e}")
    print("Run: pip install -r requirements.txt")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    """Face Recognition Model using FaceNet and Pinecone for vector storage"""
    
    def __init__(self, pinecone_api_key: str, pinecone_env: str = "us-east-1"):
        """
        Initialize the face recognition model
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_env: Pinecone environment
        """
        # Check CUDA availability and setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.info("⚠️  No GPU detected, using CPU")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.INDEX_NAME = "face-recognition-index"
        self.EMBEDDING_DIM = 512
        self.threshold = 0.55  # Lowered threshold for better recognition like model4.py
        self.feature_extractor_type = "facenet"  # For compatibility with app.py
        
        self._setup_pinecone()
        self._load_models()
        self._setup_transforms()
    
    def _setup_pinecone(self):
        """Setup Pinecone index"""
        try:
            # Create index if it doesn't exist
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.INDEX_NAME not in existing_indexes:
                self.pc.create_index(
                    name=self.INDEX_NAME,
                    dimension=self.EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"Created Pinecone index: {self.INDEX_NAME}")
            
            self.index = self.pc.Index(self.INDEX_NAME)
            logger.info("Connected to Pinecone index")
        except Exception as e:
            logger.error(f"Failed to setup Pinecone: {e}")
            raise
    
    def _load_models(self):
        """Load face detection and recognition models"""
        try:
            logger.info("Loading face recognition models...")
            
            # Face recognition model (FaceNet)
            self.facenet_model = InceptionResnetV1(
                pretrained='vggface2'
            ).eval().to(self.device)
            
            # Face detection model (YOLO) with configurable path and safe fallback
            yolo_model_path = os.getenv(
                "YOLO_MODEL_PATH",
                "/home/stark/Desktop/voice-assistant/Models/Face_Models/Face_Detect_best.pt",
            )
            yolo_strict = str(os.getenv("YOLO_STRICT", "false")).strip().lower() in ("1", "true", "yes", "on")

            if not os.path.exists(yolo_model_path):
                if yolo_strict:
                    raise FileNotFoundError(f"YOLO model not found at: {yolo_model_path} (YOLO_STRICT is enabled)")
                logger.warning(
                    f"YOLO weights not found at: {yolo_model_path}. Falling back to OpenCV face detection"
                )
                self.yolo_model = None
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            else:
                # Load YOLO model with compatibility settings for older model files
                try:
                    # First try normal loading
                    self.yolo_model = YOLO(yolo_model_path)
                    logger.info("✅ YOLO model loaded successfully (normal mode)")
                except Exception as e:
                    logger.warning(f"⚠️ Normal YOLO loading failed: {e}")
                    try:
                        # Try with warning suppression for compatibility
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # Temporarily modify torch loading behavior
                            old_load = torch.load
                            def safe_load(*args, **kwargs):
                                kwargs['weights_only'] = False
                                return old_load(*args, **kwargs)
                            torch.load = safe_load
                            
                            self.yolo_model = YOLO(yolo_model_path)
                            
                            # Restore original torch.load
                            torch.load = old_load
                            logger.info("✅ YOLO model loaded successfully (compatibility mode)")
                    except Exception as e2:
                        logger.error(f"❌ Failed to load YOLO model: {e2}")
                        if yolo_strict:
                            raise
                        # Fallback to a basic face detection using OpenCV
                        logger.warning("⚠️ Falling back to OpenCV face detection")
                        self.yolo_model = None
                        self.face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
            
            if self.yolo_model and torch.cuda.is_available():
                try:
                    self.yolo_model.to(self.device)
                    logger.info("✅ YOLO model moved to GPU")
                except:
                    logger.warning("⚠️ Could not move YOLO model to GPU, using CPU")
            
            logger.info(f"✅ Face detection model initialized")
            
            # Optimize for GPU if available
            if torch.cuda.is_available():
                # Enable optimizations for GPU
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Warm up the GPU
                logger.info("Warming up GPU...")
                dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
                with torch.no_grad():
                    _ = self.facenet_model(dummy_input)
                logger.info("✅ GPU warmed up successfully")
            
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    
    def detect_faces(self, image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[np.ndarray]:
        """
        Detect faces using YOLO model or OpenCV fallback with improved accuracy
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for face detection (lowered for better recall)
            
        Returns:
            Array of bounding boxes in format [x1, y1, x2, y2] or None if no faces
        """
        try:
            if self.yolo_model is not None:
                # Use YOLO model with improved parameters for better accuracy
                results = self.yolo_model(
                    image, 
                    conf=confidence_threshold,
                    iou=0.45,  # Non-maximum suppression threshold
                    verbose=False,
                    imgsz=640,  # Input image size for better accuracy
                    augment=True,  # Use test-time augmentation for better detection
                    agnostic_nms=False,  # Class-agnostic NMS
                    max_det=20  # Maximum detections per image
                )
                
                if not results or len(results) == 0:
                    return None
                
                # Extract bounding boxes with improved filtering
                boxes = []
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            # Get box coordinates in xyxy format
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            
                            # Calculate box area for filtering tiny detections
                            box_width = x2 - x1
                            box_height = y2 - y1
                            box_area = box_width * box_height
                            
                            # Filter by confidence and minimum size
                            if (confidence >= confidence_threshold and 
                                box_width > 20 and box_height > 20 and 
                                box_area > 400):  # Minimum face area
                                boxes.append([x1, y1, x2, y2])
                
                return np.array(boxes) if boxes else None
            
            else:
                # Fallback to OpenCV Haar Cascades with improved parameters
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization for better detection
                gray = cv2.equalizeHist(gray)
                
                # Detect faces with multiple scale factors for better accuracy
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.05,  # Smaller scale factor for better detection
                    minNeighbors=6,    # Higher neighbors for fewer false positives
                    minSize=(30, 30),
                    maxSize=(300, 300),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) == 0:
                    return None
                
                # Convert from (x, y, w, h) to (x1, y1, x2, y2) format
                boxes = []
                for (x, y, w, h) in faces:
                    boxes.append([x, y, x + w, y + h])
                
                return np.array(boxes)
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None
    
    def get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding from cropped face image
        
        Args:
            face_bgr: BGR face image as numpy array
            
        Returns:
            Face embedding or None if invalid
        """
        try:
            # Validate crop size
            if face_bgr.shape[0] < 20 or face_bgr.shape[1] < 20:
                return None
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # Preprocess and get embedding
            face_tensor = self.preprocess(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor)
                # Move to CPU immediately to free GPU memory
                embedding = embedding.cpu().numpy()[0]
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Clear GPU cache on error too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
    
    def enroll_face(self, image_path: str, face_id: str, name: str, additional_metadata: dict = None) -> bool:
        """
        Enroll a new face in the database
        
        Args:
            image_path: Path to the image file
            face_id: Unique face ID
            name: Face name/label
            additional_metadata: Additional metadata dictionary (optional)
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect faces using YOLO with improved accuracy
            boxes = self.detect_faces(img, confidence_threshold=0.3)
            if boxes is None or len(boxes) == 0:
                logger.error(f"No faces detected in {image_path}")
                return False
            
            # Process the first detected face
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding to the bounding box
            h, w = img.shape[:2]
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_crop = img[y1:y2, x1:x2]
            embedding = self.get_embedding(face_crop)
            
            if embedding is not None:
                # Prepare metadata
                metadata = {
                    "name": name, 
                    "image_path": image_path,
                    "face_id": face_id
                }
                
                # Add additional metadata if provided
                if additional_metadata:
                    metadata.update(additional_metadata)
                
                # Store in Pinecone
                self.index.upsert(vectors=[(
                    face_id, 
                    embedding.tolist(), 
                    metadata
                )])
                
                # Save face image for frontend display
                self.save_face_image(face_id, face_crop, name)
                
                logger.info(f"Enrolled {name} with ID {face_id}")
                return True
            else:
                logger.error("Failed to get embedding for face")
                return False
                
        except Exception as e:
            logger.error(f"Error enrolling face: {e}")
            return False
    
    def enroll_face_multiple(self, image_paths: list, face_id: str, name: str, additional_metadata: dict = None) -> bool:
        """
        Enroll a new face using multiple images for better accuracy
        
        Args:
            image_paths: List of paths to image files
            face_id: Unique face ID
            name: Face name/label
            additional_metadata: Additional metadata dictionary (optional)
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            logger.info(f"Processing {len(image_paths)} images for enrollment: {name}")
            
            all_embeddings = []
            successful_images = []
            
            # Process each image
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue
                
                # Detect faces using YOLO
                boxes = self.detect_faces(img, confidence_threshold=0.3)
                if boxes is None or len(boxes) == 0:
                    logger.warning(f"No faces detected in {image_path}")
                    continue
                
                # Process the first detected face
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding to the bounding box
                h, w = img.shape[:2]
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                face_crop = img[y1:y2, x1:x2]
                embedding = self.get_embedding(face_crop)
                
                if embedding is not None:
                    all_embeddings.append(embedding)
                    successful_images.append(image_path)
                    logger.info(f"✅ Successfully extracted embedding from image {i+1}")
                else:
                    logger.warning(f"Failed to get embedding from image {i+1}")
            
            if not all_embeddings:
                logger.error("No valid embeddings extracted from any image")
                return False
            
            # Calculate average embedding for better representation
            avg_embedding = np.mean(all_embeddings, axis=0)
            
            # Prepare metadata
            metadata = {
                "name": name, 
                "face_id": face_id,
                "image_count": len(successful_images),
                "successful_images": successful_images
            }
            
            # Add additional metadata if provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Store in Pinecone with the average embedding
            self.index.upsert(vectors=[(
                face_id, 
                avg_embedding.tolist(), 
                metadata
            )])
            
            # Save the best face image (first successful one) for display
            if successful_images:
                best_img = cv2.imread(successful_images[0])
                if best_img is not None:
                    boxes = self.detect_faces(best_img, confidence_threshold=0.3)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0]
                        x1, y1, x2, y2 = map(int, box)
                        h, w = best_img.shape[:2]
                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)
                        face_crop = best_img[y1:y2, x1:x2]
                        self.save_face_image(face_id, face_crop, name)
            
            logger.info(f"✅ Enrolled {name} with ID {face_id} using {len(successful_images)}/{len(image_paths)} images")
            return True
                
        except Exception as e:
            logger.error(f"Error enrolling face with multiple images: {e}")
            return False
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face from its embedding with improved accuracy
        
        Args:
            face_embedding: Face embedding vector
            
        Returns:
            Tuple of (recognized_name, confidence_score)
        """
        try:
            # Query Pinecone with enhanced parameters for better accuracy
            query_result = self.index.query(
                vector=face_embedding.tolist(), 
                top_k=3,  # Get top 3 matches for better analysis
                include_metadata=True
            )
            
            if query_result['matches'] and len(query_result['matches']) > 0:
                best_match = query_result['matches'][0]
                score = best_match['score']
                metadata = best_match.get('metadata', {})
                
                # Use a lower threshold for better recognition (like model4.py)
                recognition_threshold = 0.55  # Lowered from 0.6 for better accuracy
                
                if score > recognition_threshold:
                    name = metadata.get('name', "Unknown")
                    # Update recognition count
                    face_id = best_match['id']
                    self.update_recognition_count(face_id)
                    logger.info(f"✅ Recognized: {name} (score: {score:.3f})")
                    return name, score
                else:
                    logger.debug(f"⚠️ Low confidence match: {score:.3f} < {recognition_threshold}")
            
            return "Unknown", 0.0
            
        except Exception as e:
            logger.error(f"Error recognizing face: {e}")
            return "Unknown", 0.0
    
    def process_video(self, video_path: str, output_path: str = None, skip_frames: int = 5, resize_factor: float = 0.5) -> tuple:
        """
        Process video for face recognition with optimizations
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
            skip_frames: Process every Nth frame for speed (default: 5)
            resize_factor: Resize frame for faster processing (default: 0.5)
            
        Returns:
            Tuple of (output_path, statistics_dict)
        """
        if output_path is None:
            output_path = "recognized_output.mp4"
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate processing parameters for speed optimization
            process_width = int(width * resize_factor)
            process_height = int(height * resize_factor)
            
            # Try different codecs for better compatibility
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 codec (most compatible)
                cv2.VideoWriter_fourcc(*'H264'),  # H.264 codec
                cv2.VideoWriter_fourcc(*'avc1'),  # Another H.264 variant
                cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG (fallback)
            ]
            
            out = None
            for fourcc in fourcc_options:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    logger.info(f"✅ Using codec: {fourcc}")
                    break
                out.release()
                
            if out is None or not out.isOpened():
                logger.error("Failed to open video writer with any codec")
                raise ValueError("Could not initialize video writer")
            
            logger.info(f"🚀 OPTIMIZED Processing: {total_frames} frames on {self.device}")
            logger.info(f"📹 Video specs: {width}x{height} @ {fps}fps")
            logger.info(f"⚡ Optimizations: Skip {skip_frames} frames, Resize to {resize_factor*100}%")
            
            frame_count = 0
            processed_count = 0
            faces_detected_total = 0
            recognized_faces = set()
            last_detections = []  # Cache last detections for skipped frames
            
            with tqdm(total=total_frames, desc="🎬 Processing video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Use cached detections for skipped frames to maintain smooth output
                    if frame_count % skip_frames == 0:
                        # Resize frame for faster processing
                        small_frame = cv2.resize(frame, (process_width, process_height))
                        
                        # Detect faces using YOLO on smaller frame with improved accuracy
                        boxes = self.detect_faces(small_frame, confidence_threshold=0.3)
                        
                        # Scale boxes back to original size
                        current_detections = []
                        if boxes is not None:
                            scale_x = width / process_width
                            scale_y = height / process_height
                            
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                # Scale back to original size
                                x1 = int(x1 * scale_x)
                                y1 = int(y1 * scale_y)
                                x2 = int(x2 * scale_x)
                                y2 = int(y2 * scale_y)
                                
                                # Add padding
                                padding = 5
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(width, x2 + padding)
                                y2 = min(height, y2 + padding)
                                
                                face_crop = frame[y1:y2, x1:x2]
                                embedding = self.get_embedding(face_crop)
                                
                                if embedding is not None:
                                    faces_detected_total += 1
                                    name, score = self.recognize_face(embedding)
                                    
                                    if name != "Unknown":
                                        recognized_faces.add(name)
                                    
                                    current_detections.append({
                                        'box': (x1, y1, x2, y2),
                                        'name': name,
                                        'score': score
                                    })
                        
                        last_detections = current_detections
                        processed_count += 1
                    
                    # Draw detections (using last detections for skipped frames)
                    for detection in last_detections:
                        x1, y1, x2, y2 = detection['box']
                        name = detection['name']
                        score = detection['score']
                        
                        # Draw bounding box and label with better styling
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        thickness = 3
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Prepare label
                        label = f"{name}"
                        if name != "Unknown":
                            confidence_percent = int(score * 100)
                            label += f" ({confidence_percent}%)"
                        
                        # Calculate text size and background
                        font_scale = 0.7
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                        )
                        
                        # Draw background rectangle for text
                        text_y = y1 - 10
                        if text_y - text_height - 10 < 0:  # If too close to top, put text below
                            text_y = y2 + text_height + 10
                        
                        cv2.rectangle(
                            frame, 
                            (x1, text_y - text_height - 5), 
                            (x1 + text_width + 10, text_y + 5), 
                            color, -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            frame, label, (x1 + 5, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                        )
                    
                    out.write(frame)
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            out.release()
            
            logger.info(f"✅ SPEED OPTIMIZED: Processed {frame_count} frames in record time!")
            logger.info(f"🚀 Actually processed {processed_count} frames (skipped {frame_count - processed_count} for speed)")
            logger.info(f"📊 Statistics: {faces_detected_total} faces detected, {len(recognized_faces)} unique people")
            if recognized_faces:
                logger.info(f"👥 Recognized faces: {', '.join(recognized_faces)}")
            
            return output_path, {
                'total_frames': frame_count,
                'processed_frames': processed_count,
                'faces_detected': faces_detected_total,
                'unique_people': len(recognized_faces),  # Keep for API compatibility
                'recognized_names': list(recognized_faces),
                'optimization': f"Processed every {skip_frames} frames at {resize_factor*100}% size"
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
    
    def process_image(self, image_path: str, output_path: str = None) -> tuple:
        """
        Process single image for face recognition
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            
        Returns:
            Tuple of (output_path, statistics_dict)
        """
        if output_path is None:
            output_path = "recognized_image.jpg"
        
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            faces_detected = 0
            recognized_faces = set()
            
            # Detect faces using YOLO with improved accuracy
            boxes = self.detect_faces(img, confidence_threshold=0.3)
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Add padding
                    h, w = img.shape[:2]
                    padding = 10
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    face_crop = img[y1:y2, x1:x2]
                    embedding = self.get_embedding(face_crop)
                    
                    if embedding is not None:
                        faces_detected += 1
                        name, score = self.recognize_face(embedding)
                        
                        if name != "Unknown":
                            recognized_faces.add(name)
                        
                        # Draw bounding box and label with better styling
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        thickness = 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                        
                        # Prepare label
                        label = f"{name}"
                        if name != "Unknown":
                            confidence_percent = int(score * 100)
                            label += f" ({confidence_percent}%)"
                        
                        # Calculate text size and background
                        font_scale = 0.9
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                        )
                        
                        # Draw background rectangle for text
                        text_y = y1 - 15
                        if text_y - text_height - 10 < 0:  # If too close to top, put text below
                            text_y = y2 + text_height + 20
                        
                        cv2.rectangle(
                            img, 
                            (x1, text_y - text_height - 8), 
                            (x1 + text_width + 15, text_y + 8), 
                            color, -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            img, label, (x1 + 8, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
                        )
            
            cv2.imwrite(output_path, img)
            logger.info(f"✅ Processed image saved to: {output_path}")
            logger.info(f"📊 Statistics: {faces_detected} faces detected, {len(recognized_faces)} faces recognized")
            if recognized_faces:
                logger.info(f"👥 Recognized faces: {', '.join(recognized_faces)}")
            
            return output_path, {
                'faces_detected': faces_detected,
                'unique_people': len(recognized_faces),  # Keep for API compatibility
                'recognized_names': list(recognized_faces)
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def get_enrolled_faces(self) -> Dict[str, Any]:
        """
        Get list of enrolled faces with metadata
        
        Returns:
            Dictionary containing face data and statistics
        """
        try:
            # Create faces directory if it doesn't exist
            faces_dir = Path("enrolled_faces")
            faces_dir.mkdir(exist_ok=True)
            
            # Get face images and metadata
            faces_list = []
            metadata_file = faces_dir / "metadata.json"
            
            # Load existing metadata
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    metadata = {}
            
            # Scan for face images
            for img_file in faces_dir.glob("*.jpg"):
                face_id = img_file.stem
                
                # Try to get additional metadata from Pinecone
                pinecone_metadata = {}
                try:
                    query_result = self.index.query(id=face_id, top_k=1, include_metadata=True)
                    if query_result.matches:
                        pinecone_metadata = query_result.matches[0].metadata
                except:
                    pass
                
                face_data = {
                    "id": face_id,
                    "name": metadata.get(face_id, {}).get("name", pinecone_metadata.get("name", f"Face_{face_id[:8]}")),
                    "image_path": str(img_file),
                    "enrolled_date": metadata.get(face_id, {}).get("enrolled_date", pinecone_metadata.get("enrollment_date", "Unknown")),
                    "confidence_threshold": metadata.get(face_id, {}).get("confidence_threshold", 0.8),
                    "times_recognized": metadata.get(face_id, {}).get("times_recognized", 0),
                    # Additional metadata from Pinecone
                    "person_type": pinecone_metadata.get("person_type", "Unknown"),
                    "employee_id": pinecone_metadata.get("employee_id", ""),
                    "position": pinecone_metadata.get("position", ""),
                    "department": pinecone_metadata.get("department", ""),
                    "phone_number": pinecone_metadata.get("phone_number", ""),
                    "special_notes": pinecone_metadata.get("special_notes", ""),
                    "image_count": pinecone_metadata.get("image_count", 1)
                }
                faces_list.append(face_data)
            
            # Get index statistics
            stats = self.index.describe_index_stats()
            
            return {
                "faces": faces_list,
                "total_faces": len(faces_list),
                "vector_count": stats.total_vector_count,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting enrolled faces: {e}")
            return {"faces": [], "total_faces": 0, "vector_count": 0}
    
    def delete_face(self, face_id: str) -> bool:
        """
        Delete a face from the database and remove associated files
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Delete from Pinecone index
            self.index.delete(ids=[face_id])
            
            # Delete image file and update metadata
            faces_dir = Path("enrolled_faces")
            img_file = faces_dir / f"{face_id}.jpg"
            if img_file.exists():
                img_file.unlink()
            
            # Update metadata
            metadata_file = faces_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    if face_id in metadata:
                        del metadata[face_id]
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                except:
                    pass
            
            logger.info(f"Deleted face with ID: {face_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False

    def update_face_metadata(self, face_id: str, name: str = None, confidence_threshold: float = None) -> bool:
        """
        Update face metadata
        
        Args:
            face_id: Face ID to update
            name: New name for the face
            confidence_threshold: New confidence threshold
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            faces_dir = Path("enrolled_faces")
            faces_dir.mkdir(exist_ok=True)
            metadata_file = faces_dir / "metadata.json"
            
            # Load existing metadata
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    metadata = {}
            
            # Update metadata
            if face_id not in metadata:
                metadata[face_id] = {}
            
            if name is not None:
                metadata[face_id]["name"] = name
            if confidence_threshold is not None:
                metadata[face_id]["confidence_threshold"] = confidence_threshold
            
            metadata[face_id]["last_updated"] = datetime.now().isoformat()
            
            # Save metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Updated metadata for face ID: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating face metadata: {e}")
            return False

    def save_face_image(self, face_id: str, face_image: np.ndarray, name: str = None) -> bool:
        """
        Save face image to enrolled faces directory
        
        Args:
            face_id: Face ID
            face_image: Face image as numpy array
            name: Face name/label
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            faces_dir = Path("enrolled_faces")
            faces_dir.mkdir(exist_ok=True)
            
            # Save image
            img_path = faces_dir / f"{face_id}.jpg"
            cv2.imwrite(str(img_path), face_image)
            
            # Update metadata
            metadata_file = faces_dir / "metadata.json"
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    metadata = {}
            
            if face_id not in metadata:
                metadata[face_id] = {
                    "enrolled_date": datetime.now().isoformat(),
                    "times_recognized": 0,
                    "confidence_threshold": 0.8
                }
            
            if name:
                metadata[face_id]["name"] = name
            
            metadata[face_id]["last_updated"] = datetime.now().isoformat()
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved face image for face ID: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving face image: {e}")
            return False

    def assess_face_quality(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Assess the quality of a face image for better recognition
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate blur (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray)
            
            # Calculate image resolution quality
            height, width = face_image.shape[:2]
            resolution_score = height * width
            
            # Overall quality score (normalized)
            quality_score = (
                min(blur_score / 100, 1.0) * 0.4 +  # Blur weight
                min(abs(brightness - 128) / 128, 1.0) * 0.2 +  # Brightness weight
                min(contrast / 50, 1.0) * 0.2 +  # Contrast weight
                min(resolution_score / 10000, 1.0) * 0.2  # Resolution weight
            )
            
            return {
                'blur_score': float(blur_score),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'resolution': [int(width), int(height)],
                'quality_score': float(quality_score),
                'is_good_quality': quality_score > 0.6
            }
            
        except Exception as e:
            logger.error(f"Error assessing face quality: {e}")
            return {
                'blur_score': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'resolution': [0, 0],
                'quality_score': 0.0,
                'is_good_quality': False
            }

    def get_face_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks for pose estimation
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Facial landmarks array or None
        """
        try:
            # This is a placeholder for facial landmark detection
            # You can integrate dlib or mediapipe for better landmark detection
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # For now, return basic face center and estimated landmarks
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            
            # Estimated landmark positions (eyes, nose, mouth)
            landmarks = np.array([
                [center_x - w//4, center_y - h//4],  # Left eye
                [center_x + w//4, center_y - h//4],  # Right eye
                [center_x, center_y],                # Nose
                [center_x - w//6, center_y + h//4], # Left mouth corner
                [center_x + w//6, center_y + h//4], # Right mouth corner
            ])
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None

    def detect_face_attributes(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect basic face attributes like estimated age, gender, emotion
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Dictionary with estimated attributes
        """
        try:
            # This is a basic implementation
            # For production, integrate with specialized models
            
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Basic heuristics (placeholder implementation)
            brightness = np.mean(gray)
            
            # Estimate based on image characteristics
            estimated_age = "Unknown"
            estimated_gender = "Unknown"
            estimated_emotion = "Neutral"
            
            # Simple brightness-based emotion estimation
            if brightness > 150:
                estimated_emotion = "Happy"
            elif brightness < 100:
                estimated_emotion = "Serious"
            
            return {
                'estimated_age': estimated_age,
                'estimated_gender': estimated_gender,
                'estimated_emotion': estimated_emotion,
                'confidence': 0.5  # Low confidence for basic implementation
            }
            
        except Exception as e:
            logger.error(f"Error detecting face attributes: {e}")
            return {
                'estimated_age': "Unknown",
                'estimated_gender': "Unknown", 
                'estimated_emotion': "Unknown",
                'confidence': 0.0
            }

    def update_recognition_count(self, face_id: str) -> bool:
        """
        Update the recognition count for a face
        
        Args:
            face_id: Face ID to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            faces_dir = Path("enrolled_faces")
            metadata_file = faces_dir / "metadata.json"
            
            if not metadata_file.exists():
                return False
            
            # Load metadata
            metadata = {}
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                return False
            
            if face_id in metadata:
                # Increment recognition count
                metadata[face_id]["times_recognized"] = metadata[face_id].get("times_recognized", 0) + 1
                metadata[face_id]["last_recognized"] = datetime.now().isoformat()
                
                # Save updated metadata
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating recognition count: {e}")
            return False