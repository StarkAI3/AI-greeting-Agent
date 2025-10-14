import os
import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model3 import FaceRecognitionModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enroll_app")


face_model: Optional[FaceRecognitionModel] = None


def ensure_dirs():
    for folder in ["uploads", "outputs", "static", "enrolled_faces"]:
        Path(folder).mkdir(exist_ok=True)


def safe_filename(name: str) -> str:
    keep = "-.()_"
    return "".join(c if c.isalnum() or c in keep else "_" for c in name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_model
    ensure_dirs()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.warning("PINECONE_API_KEY not set! Enrollment will fail.")
        face_model = None
    else:
        try:
            face_model = FaceRecognitionModel(pinecone_api_key=api_key)
            logger.info("Face model initialized for enrollment app")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            face_model = None
    yield


ensure_dirs()  # ensure static directories exist before mounting
app = FastAPI(title="Face Enrollment Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/enrolled_faces", StaticFiles(directory="enrolled_faces"), name="enrolled_faces")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path("static") / "enroll.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse(
        """
        <!doctype html>
        <html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
        <title>Face Enrollment</title></head>
        <body style=\"font-family: sans-serif; max-width: 900px; margin: 20px auto;\">
        <h2>👤 Enroll New Face</h2>
        <form id=\"f\">
          <div><label>Name</label><br/>
            <input name=\"faceName\" placeholder=\"Enter person's full name\" required>
          </div>
          <div><label>Face ID</label><br/>
            <input name=\"faceId\" placeholder=\"Enter unique face ID (e.g., FACE001)\" required>
          </div>
          <div><label>Employee ID</label><br/>
            <input name=\"employeeId\" placeholder=\"Enter employee ID (optional)\">
          </div>
          <div><label>Position</label><br/>
            <input name=\"position\" placeholder=\"Enter job title or position (optional)\">
          </div>
          <div><label>Department</label><br/>
            <input name=\"department\" placeholder=\"Enter department (optional)\">
          </div>
          <div><label>Date of Birth</label><br/>
            <input name=\"dateOfBirth\" placeholder=\"dd/mm/yyyy\">
          </div>
          <div><label>Joining Date</label><br/>
            <input name=\"joiningDate\" placeholder=\"dd/mm/yyyy\">
          </div>
          <div><label>Phone Number</label><br/>
            <input name=\"phoneNumber\" placeholder=\"Enter phone number (optional)\">
          </div>
          <div><label>Who are they?</label><br/>
            <select name=\"personType\" required>
              <option value=\"employee\">Employee</option>
              <option value=\"visitor\">Visitor</option>
              <option value=\"guest\">Guest</option>
            </select>
          </div>
          <div><label>Special Notes (Optional)</label><br/>
            <input name=\"specialNotes\" placeholder=\"e.g., VIP client, first time visitor, delivery person\">
          </div>
          <div><label>Purpose of Visit (Optional)</label><br/>
            <input name=\"purposeOfVisit\" placeholder=\"Enter purpose of visit (optional)\">
          </div>
          <div><label>Current Project</label><br/>
            <input name=\"currentProject\" placeholder=\"e.g., AI Development, Marketing Campaign...\">
          </div>
          <div><label>Team Size</label><br/>
            <input name=\"teamSize\" placeholder=\"Number of team members\" type=\"number\" min=\"0\">
          </div>
          <div><label>Office Location</label><br/>
            <input name=\"officeLocation\" placeholder=\"e.g., Floor 3, Building A, Mumbai...\">
          </div>
          <div><label>Work Schedule</label><br/>
            <input name=\"workSchedule\" placeholder=\"Morning (10 AM - 7 PM)\">
          </div>
          <div><label>Skills (comma-separated)</label><br/>
            <input name=\"skills\" placeholder=\"e.g., Python, Machine Learning, Leadership...\">
          </div>
          <div><label>Interests (comma-separated)</label><br/>
            <input name=\"interests\" placeholder=\"e.g., Music, Sports, Technology, Travel...\">
          </div>
          <div><label>Emergency Contact</label><br/>
            <input name=\"emergencyContact\" placeholder=\"Emergency contact number\">
          </div>
          <hr/>
          <div><label>Images (up to 5)</label><br/><input name=\"enrollImages\" type=\"file\" accept=\"image/*\" multiple required></div>
          <button>Enroll</button>
        </form>
        <pre id=\"out\"></pre>
        <script>
        document.getElementById('f').onsubmit = async (e)=>{
          e.preventDefault();
          const fd = new FormData(e.target);
          const r = await fetch('/api/enroll', {method:'POST', body: fd});
          const j = await r.json();
          document.getElementById('out').textContent = JSON.stringify(j, null, 2);
        };
        </script>
        </body></html>
        """
    )


@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "healthy",
        "model_loaded": face_model is not None,
        "gpu": torch.cuda.is_available(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/enroll")
async def enroll(
    faceName: str = Form(...),
    faceId: str = Form(...),
    personType: str = Form(...),
    employeeId: str = Form("") ,
    position: str = Form(""),
    department: str = Form(""),
    dateOfBirth: str = Form(""),
    joiningDate: str = Form(""),
    phoneNumber: str = Form(""),
    specialNotes: str = Form(""),
    purposeOfVisit: str = Form(""),
    currentProject: str = Form(""),
    teamSize: str = Form(""),
    officeLocation: str = Form(""),
    workSchedule: str = Form(""),
    skills: str = Form(""),
    interests: str = Form(""),
    emergencyContact: str = Form(""),
    enrollImages: List[UploadFile] = File(...),
):
    if face_model is None:
        return JSONResponse({"success": False, "message": "Model not initialized"}, status_code=500)

    files = enrollImages or []
    if len(files) == 0:
        return JSONResponse({"success": False, "message": "No images provided"}, status_code=400)
    if len(files) > 5:
        return JSONResponse({"success": False, "message": "Maximum 5 images allowed"}, status_code=400)

    # Save files
    saved_paths: List[str] = []
    try:
        for i, uf in enumerate(files):
            ext = os.path.splitext(uf.filename or "")[1].lower() or ".jpg"
            fname = safe_filename(f"{faceId}_{i}_{uuid.uuid4().hex}{ext}")
            fpath = os.path.join("uploads", fname)
            with open(fpath, "wb") as w:
                data = await uf.read()
                w.write(data)
            saved_paths.append(fpath)

        metadata = {
            "person_type": personType,
            "employee_id": employeeId,
            "position": position,
            "department": department,
            "date_of_birth": dateOfBirth,
            "joining_date": joiningDate,
            "phone_number": phoneNumber,
            "special_notes": specialNotes,
            "purpose_of_visit": purposeOfVisit,
            "current_project": currentProject,
            "team_size": teamSize,
            "office_location": officeLocation,
            "work_schedule": workSchedule,
            "skills": [s.strip() for s in skills.split(',') if s.strip()] if skills else [],
            "interests": [s.strip() for s in interests.split(',') if s.strip()] if interests else [],
            "emergency_contact": emergencyContact,
            "enrollment_date": datetime.now().isoformat(),
            "image_count": len(saved_paths),
            "image_paths": saved_paths,
        }

        ok = face_model.enroll_face_multiple(saved_paths, faceId, faceName, metadata)
        if not ok:
            for p in saved_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            return JSONResponse({"success": False, "message": "No faces detected or enrollment failed"}, status_code=400)

        return {
            "success": True,
            "message": f"Successfully enrolled {faceName} ({personType}) with {len(saved_paths)} images",
            "face_id": faceId,
            "face_name": faceName,
            "person_type": personType,
            "employee_id": employeeId,
            "department": department,
            "image_count": len(saved_paths),
        }
    except Exception as e:
        logger.error(f"Enroll error: {e}")
        return JSONResponse({"success": False, "message": f"Internal error: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("Enrollment app running on http://localhost:6002")
    uvicorn.run(app, host="0.0.0.0", port=6002)


