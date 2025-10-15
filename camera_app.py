import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import cv2
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Local model
from model3 import FaceRecognitionModel
# Greeting service is optional; guard import so server still starts if module is missing
try:
    from greeting_service import GreetingManager  # type: ignore
except Exception:
    GreetingManager = None  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camera_app")


# Globals
face_model = None
current_model_type = "enhanced"
camera_active = False
camera_cap = None
greeting_manager = None


def get_current_model():
    return face_model


def ensure_dirs():
    for folder in ["uploads", "outputs", "static", "enrolled_faces"]:
        Path(folder).mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_model, greeting_manager
    ensure_dirs()
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    # Initialize face recognition model
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning("PINECONE_API_KEY not set, running camera without recognition model")
        face_model = None
    else:
        try:
            face_model = FaceRecognitionModel(pinecone_api_key=api_key)
            logger.info("Face model initialized for camera app")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            face_model = None
    
    # Initialize Greeting service (Gemini + Sarvam AI + ElevenLabs)
    try:
        gemini_key = os.getenv("GEMINI_API_KEY")
        sarvam_key = os.getenv("SARVAM_API_KEY")
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        
        if GreetingManager and gemini_key and (sarvam_key or elevenlabs_key):
            greeting_manager = GreetingManager(
                gemini_api_key=gemini_key,
                sarvam_api_key=sarvam_key,
                elevenlabs_api_key=elevenlabs_key,
                cooldown_minutes=5,
                default_tts_provider="sarvam" if sarvam_key else "elevenlabs"
            )
            logger.info("✅ Greeting service initialized (Gemini + Multi-TTS)")
        else:
            if not gemini_key:
                logger.warning("⚠️ GEMINI_API_KEY not set in .env file")
            elif not (sarvam_key or elevenlabs_key):
                logger.warning("⚠️ SARVAM_API_KEY or ELEVENLABS_API_KEY not set in .env file")
            else:
                logger.warning("Greeting service unavailable (greeting_service.py missing)")
            greeting_manager = None
    except Exception as e:
        logger.error(f"Failed to initialize greeting service: {e}")
        greeting_manager = None

    yield

    # Cleanup camera on shutdown
    global camera_active, camera_cap
    camera_active = False
    if camera_cap is not None:
        try:
            camera_cap.release()
        except Exception:
            pass
    
    # Cleanup greeting service
    if greeting_manager:
        try:
            greeting_manager.cleanup()
        except Exception:
            pass


app = FastAPI(title="Real-time Camera Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static") / "camera.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    # Fallback minimal UI
    return HTMLResponse(
        """
        <!doctype html>
        <html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
        <title>Camera Recognition</title></head>
        <body style=\"font-family: sans-serif; max-width: 900px; margin: 20px auto;\">
        <h2>📹 Real-time Camera Recognition</h2>
        <div id=\"status\">Idle</div>
        <div style=\"margin: 10px 0;\">
            <button onclick=\"startCam()\">Start</button>
            <button onclick=\"stopCam()\">Stop</button>
        </div>
        <img id=\"stream\" style=\"width:100%;max-width:800px;border:2px solid #ddd;border-radius:8px\" />
        <script>
        async function startCam(){
          document.getElementById('status').innerText = 'Starting...';
          const r = await fetch('/api/camera/start', {method:'POST'});
          const j = await r.json();
          if(j.success){
            document.getElementById('status').innerText = 'Running';
            document.getElementById('stream').src = '/api/camera/stream?t=' + Date.now();
          }else{
            document.getElementById('status').innerText = 'Error: ' + j.message;
          }
        }
        async function stopCam(){
          await fetch('/api/camera/stop', {method:'POST'});
          document.getElementById('status').innerText = 'Stopped';
          document.getElementById('stream').src = '';
        }
        </script>
        </body></html>
        """
    )


@app.get("/api/health")
async def health():
    import torch
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        }
    else:
        gpu_info = {"gpu_available": False, "device": "CPU"}

    return {
        "status": "healthy",
        "model_loaded": get_current_model() is not None,
        "model_type": "YOLO + FaceNet",
        "current_model": current_model_type,
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


@app.post("/api/camera/start")
async def start_camera():
    global camera_active, camera_cap
    if camera_active:
        return JSONResponse({"success": False, "message": "Camera already running"}, status_code=400)

    # Test access
    test = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not test.isOpened():
        test.release()
        return JSONResponse({"success": False, "message": "Cannot access camera"}, status_code=400)
    test.release()

    camera_active = True
    return {"success": True, "message": "Camera stream started", "stream_url": "/api/camera/stream"}


@app.post("/api/camera/stop")
async def stop_camera():
    global camera_active, camera_cap
    camera_active = False
    if camera_cap is not None:
        try:
            camera_cap.release()
        except Exception:
            pass
        camera_cap = None
    return {"success": True, "message": "Camera stream stopped"}


def _frame_generator():
    global camera_active, camera_cap, greeting_manager

    # Try backends suitable for Linux first
    for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            camera_cap = cap
            logger.info(f"Camera opened with backend: {backend}")
            break
        cap.release()
    else:
        # yield a text frame on failure
        yield (b"--frame\r\n"
               b"Content-Type: text/plain\r\n\r\n"
               b"Camera access failed\r\n")
        return

    camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    process_every_n = 3
    last_faces = []

    try:
        while camera_active:
            ok, frame = camera_cap.read()
            if not ok:
                break
            frame_count += 1

            model = get_current_model()
            if model is not None and frame_count % process_every_n == 0:
                try:
                    boxes = model.detect_faces(frame, confidence_threshold=0.3)
                    cur = []
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            h, w = frame.shape[:2]
                            pad = 20
                            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                                continue
                            embedding = model.get_embedding(face_crop)
                            name = "Unknown"; score = 0.0
                            if embedding is not None:
                                try:
                                    res = model.index.query(vector=embedding.tolist(), top_k=1, include_metadata=True)
                                    if res.matches:
                                        m = res.matches[0]
                                        score = m.score
                                        if score > 0.7:
                                            name = m.metadata.get('name', 'Unknown')
                                            
                                            # 🎤 Greet the recognized person with full details (Gemini + Sarvam AI)
                                            if greeting_manager:
                                                try:
                                                    # Extract all person details from metadata
                                                    person_details = {
                                                        'position': m.metadata.get('position', ''),
                                                        'department': m.metadata.get('department', ''),
                                                        'current_project': m.metadata.get('current_project', ''),
                                                        'interests': m.metadata.get('interests', []),
                                                        'skills': m.metadata.get('skills', []),
                                                        'office_location': m.metadata.get('office_location', ''),
                                                        'work_schedule': m.metadata.get('work_schedule', ''),
                                                        'team_size': m.metadata.get('team_size', ''),
                                                        'person_type': m.metadata.get('person_type', 'employee'),
                                                        'special_notes': m.metadata.get('special_notes', ''),
                                                        'employee_id': m.metadata.get('employee_id', ''),
                                                        'date_of_birth': m.metadata.get('date_of_birth', ''),
                                                        'joining_date': m.metadata.get('joining_date', ''),
                                                        'phone_number': m.metadata.get('phone_number', ''),
                                                        'purpose_of_visit': m.metadata.get('purpose_of_visit', '')
                                                    }
                                                    greeting_manager.greet_if_needed(name, score, person_details)
                                                except Exception as greet_err:
                                                    logger.error(f"Greeting error: {greet_err}")
                                except Exception:
                                    pass
                            cur.append({"bbox": (x1, y1, x2, y2), "name": name, "score": score})
                    last_faces = cur
                except Exception as e:
                    logger.error(f"recognition error: {e}")

            for f in last_faces:
                x1, y1, x2, y2 = f["bbox"]
                name = f["name"]; score = f["score"]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = name if name == "Unknown" else f"{name} ({score:.2f})"
                cv2.putText(frame, label, (x1+5, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(frame, f"LIVE - Frame {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            ok, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)
    finally:
        if camera_cap is not None:
            try:
                camera_cap.release()
            except Exception:
                pass


@app.get("/api/camera/stream")
async def camera_stream():
    return StreamingResponse(_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/greeting/stats")
async def greeting_stats():
    """Get greeting statistics"""
    if greeting_manager:
        return greeting_manager.get_stats()
    return {"error": "Greeting manager not initialized"}


@app.post("/api/greeting/reset")
async def reset_greeting_cooldown(name: str = None):
    """Reset greeting cooldown for a person or all people"""
    if greeting_manager:
        greeting_manager.reset_cooldown(name)
        return {"success": True, "message": f"Cooldown reset for {name or 'all people'}"}
    return JSONResponse({"success": False, "message": "Greeting manager not initialized"}, status_code=500)


@app.post("/api/greeting/test")
async def test_greeting(
    name: str = "Test User",
    position: str = "Software Developer",
    department: str = "Engineering",
    interests: str = "AI, Music, Travel"
):
    """Test the greeting system with sample person details"""
    if greeting_manager:
        try:
            # Create sample person details for testing
            person_details = {
                'position': position,
                'department': department,
                'current_project': 'Face Recognition System',
                'interests': [i.strip() for i in interests.split(',')],
                'skills': ['Python', 'Machine Learning', 'Computer Vision'],
                'person_type': 'employee',
                'special_notes': 'Always brings positive energy to the team'
            }
            success = greeting_manager.greet_person(name, 1.0, person_details)
            return {"success": success, "message": f"Test greeting for {name} with full details"}
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    return JSONResponse({"success": False, "message": "Greeting service not initialized"}, status_code=500)


@app.post("/api/greeting/config")
async def update_greeting_config(
    speaker: str = None,
    pace: float = None,
    language: str = None,
    pitch: float = None,
    loudness: float = None,
    speech_sample_rate: int = None,
):
    """Update Sarvam AI TTS voice configuration"""
    if greeting_manager:
        try:
            greeting_manager.tts_service.set_voice_config(
                speaker=speaker,
                pace=pace,
                language=language,
                pitch=pitch,
                loudness=loudness,
                speech_sample_rate=speech_sample_rate,
            )
            return {"success": True, "message": "Voice configuration updated", 
                    "config": greeting_manager.tts_service.default_config}
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    return JSONResponse({"success": False, "message": "Greeting service not initialized"}, status_code=500)


@app.get("/api/voice/available")
async def get_available_voices():
    """Get list of available TTS voices"""
    if greeting_manager:
        try:
            return greeting_manager.get_available_voices()
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    return JSONResponse({"success": False, "message": "Greeting service not initialized"}, status_code=500)


@app.post("/api/voice/set")
async def set_voice_provider(provider: str):
    """
    Switch between TTS providers
    
    Args:
        provider: "sarvam" or "elevenlabs"
    """
    if greeting_manager:
        try:
            success = greeting_manager.set_tts_provider(provider)
            if success:
                return {
                    "success": True, 
                    "message": f"Switched to {provider} TTS",
                    "current_provider": greeting_manager.get_tts_provider()
                }
            else:
                return JSONResponse(
                    {"success": False, "message": f"Failed to switch to {provider} - service not available"},
                    status_code=400
                )
        except Exception as e:
            return JSONResponse({"success": False, "message": str(e)}, status_code=500)
    return JSONResponse({"success": False, "message": "Greeting service not initialized"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("Camera app running on http://localhost:6001")
    uvicorn.run(app, host="0.0.0.0", port=6001)


