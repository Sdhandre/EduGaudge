from fastapi import FastAPI, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, joblib
import mediapipe as mp
from ultralytics import YOLO
from fastapi import HTTPException

# MongoDB imports
import urllib.parse
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import asyncio

# ADD GEMINI AI IMPORTS
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="EduGauge WebSocket Backend")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIGURE GEMINI AI (IMPROVED WITH ERROR HANDLING)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Move to .env later
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Try multiple models to avoid quota issues
    models_to_try = [
        "models/gemini-2.0-flash-lite",      # Lightest model first
        "models/gemini-flash-lite-latest",   # Stable lite version
        "models/gemini-2.5-flash-lite",     # Latest lite
        "models/gemini-2.0-flash",          # Standard if lite unavailable
        "models/gemini-2.5-flash"           # Fallback
    ]
    
    gemini_model = None
    working_model_name = None
    
    for model_name in models_to_try:
        try:
            print(f"🧪 Testing {model_name}...")
            test_model = genai.GenerativeModel(model_name)
            # Quick test with minimal token usage
            test_response = test_model.generate_content("Hi")
            gemini_model = test_model
            working_model_name = model_name
            print(f"✅ Gemini AI configured with: {model_name}")
            break
        except Exception as e:
            if "quota" in str(e).lower():
                print(f"⚠️ {model_name} - Quota exceeded, trying next...")
                continue
            else:
                print(f"❌ {model_name} - Failed: {str(e)[:50]}...")
                continue
    
    if not gemini_model:
        print("❌ All Gemini models failed or quota exceeded")
        gemini_model = None
else:
    gemini_model = None
    working_model_name = None
    print("⚠️ Gemini API key not found")

# Rate limiting for AI requests
last_ai_request = datetime.min
AI_REQUEST_DELAY = 3  # 3 seconds between requests

# MongoDB setup with proper URL encoding

MONGODB_URI = os.getenv("MONGODB_URI")



# MongoDB connection state
class DatabaseState:
    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
    
    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(MONGODB_URI)
            self.db = self.client["edugauge"]
            await self.db.command("ping")
            self.connected = True
            print("✅ Connected to MongoDB successfully")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self.connected = False
            self.client = None
            self.db = None

# Create database state instance
db_state = DatabaseState()

# Load models
print("Loading models...")
clf = joblib.load("attention_model_6.pkl")

# Get feature names directly from model if available
if hasattr(clf, "feature_names_in_"):
    feature_names = clf.feature_names_in_.tolist()
    print(f"✅ Feature names loaded from model: {len(feature_names)} features")
else:
    # Fallback: generate names for 468 landmarks × 3 coordinates = 1404
    feature_names = [f"landmark_{i//3}_{['x','y','z'][i%3]}" for i in range(1404)]
    print(f"⚠️ Model has no feature names, generated {len(feature_names)} instead")

yolo_model = YOLO("yolov8n.pt")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=6, min_detection_confidence=0.7, min_tracking_confidence=0.7)

print("✅ All models loaded successfully")

# Simple tracker
class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}

    def update(self, rects):
        tracked = {}
        for r in rects:
            self.objects[self.next_id] = r
            tracked[self.next_id] = r
            self.next_id += 1
        return tracked

tracker = SimpleTracker()

# Helper function to check if two boxes overlap
def overlaps(boxA, boxB):
    x1, y1, x2, y2 = boxA
    X1, Y1, X2, Y2 = boxB
    return not (x2 < X1 or X2 < x1 or y2 < Y1 or Y2 < y1)

def extract_enhanced_features(landmarks_list):
    """Extract smart features instead of raw coordinates"""
    
    # Convert landmarks to numpy array
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list])
    
    features = []
    
    # 1. EYE ASPECT RATIOS (Most important for drowsiness)
    try:
        left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
        right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
        
        def calculate_ear(eye_points):
            """Calculate Eye Aspect Ratio"""
            try:
                v1 = np.linalg.norm(eye_points[1] - eye_points[5])
                v2 = np.linalg.norm(eye_points[2] - eye_points[4])
                h = np.linalg.norm(eye_points[0] - eye_points[3])
                return (v1 + v2) / (2.0 * h) if h > 0 else 0
            except:
                return 0.25  # Default value
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        features.extend([left_ear, right_ear, avg_ear])
    except:
        features.extend([0.25, 0.25, 0.25])  # Default values
    
    # 2. MOUTH OPENING RATIO
    try:
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]
        mouth_left = landmarks[78]
        mouth_right = landmarks[308]
        
        mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0
    except:
        mouth_ratio = 0
    
    features.append(mouth_ratio)
    
    # 3. HEAD POSE
    try:
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        head_tilt = abs(np.arctan2(right_eye_corner[1] - left_eye_corner[1], 
                                  right_eye_corner[0] - left_eye_corner[0]))
    except:
        head_tilt = 0
    
    features.append(head_tilt)
    
    return np.array(features)

# --- TEMPORAL SMOOTHING CLASS ---
class AttentionSmoothing:
    """Smooth attention detection over time"""
    def __init__(self):
        self.history = {}  # Store history for each person
        self.window_size = 10
        
    def update_person(self, person_id, is_drowsy, is_phone):
        """Update history for a person and return smoothed result"""
        if person_id not in self.history:
            self.history[person_id] = {
                'drowsy_frames': [],
                'phone_frames': []
            }
        
        # Add current frame
        self.history[person_id]['drowsy_frames'].append(is_drowsy)
        self.history[person_id]['phone_frames'].append(is_phone)
        
        # Keep only recent frames
        if len(self.history[person_id]['drowsy_frames']) > self.window_size:
            self.history[person_id]['drowsy_frames'].pop(0)
        if len(self.history[person_id]['phone_frames']) > self.window_size:
            self.history[person_id]['phone_frames'].pop(0)
        
        # Calculate smoothed results
        drowsy_count = sum(self.history[person_id]['drowsy_frames'])
        phone_count = sum(self.history[person_id]['phone_frames'])
        
        # Decision thresholds
        total_frames = len(self.history[person_id]['drowsy_frames'])
        drowsy_threshold = total_frames * 0.6  # 60% of frames
        phone_threshold = total_frames * 0.3   # 30% of frames
        
        if phone_count >= phone_threshold:
            return "USING PHONE"
        elif drowsy_count >= drowsy_threshold:
            return "DROWSY"
        else:
            return "ATTENTIVE"

# Create global smoother instance
attention_smoother = AttentionSmoothing()


def analyze_frame(frame):
    """Enhanced frame analysis with better accuracy"""
    H, W = frame.shape[:2]
    result = {"overall_score": 0, "persons": []}

    # Detect phones (same as before)
    phone_boxes = []
    for res in yolo_model(frame, verbose=False):
        for box in res.boxes:
            if int(box.cls) == 67:  # COCO class 67 = 'cell phone'
                phone_boxes.append(tuple(map(int, box.xyxy[0].tolist())))

    # Detect faces (same as before)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh_results = face_mesh.process(rgb)

    rects, landmarks_list = [], []
    if mesh_results.multi_face_landmarks:
        for lm_set in mesh_results.multi_face_landmarks:
            x_min, y_min, x_max, y_max = W, H, 0, 0
            for lm in lm_set.landmark:
                cx, cy = int(lm.x * W), int(lm.y * H)
                x_min, y_min, x_max, y_max = min(x_min, cx), min(y_min, cy), max(x_max, cx), max(y_max, cy)
            rects.append((x_min, y_min, x_max, y_max))
            landmarks_list.append(lm_set)

    # Track faces (using your existing tracker)
    tracked = tracker.update(rects)
    scores = []

    for i, (obj_id, bbox) in enumerate(tracked.items()):
        status, score = "ATTENTIVE", 90

        if i < len(landmarks_list):
            lm_set = landmarks_list[i]
            
            # --- ENHANCED DETECTION LOGIC ---
            try:
                # Method 1: Try enhanced feature detection
                enhanced_features = extract_enhanced_features(lm_set.landmark)
                
                # Enhanced drowsiness rules
                avg_ear = enhanced_features[2]
                mouth_ratio = enhanced_features[3]
                head_tilt = enhanced_features[4]
                
                # Smart drowsiness detection with multiple conditions
                is_drowsy = (avg_ear < 0.23) or (mouth_ratio > 0.7) or (head_tilt > 0.5)
                
            except:
                # Method 2: Fallback to original model
                try:
                    row = [v for lm in lm_set.landmark for v in (lm.x, lm.y, lm.z)]
                    X = np.array(row).reshape(1, -1)
                    drowsy = clf.predict(X)[0]
                    is_drowsy = (drowsy == 1)
                except:
                    # Method 3: Simple fallback
                    is_drowsy = False

            # Check phone usage (same logic as before)
            is_phone = any(overlaps(bbox, pb) for pb in phone_boxes)
            
            # --- TEMPORAL SMOOTHING ---
            smoothed_status = attention_smoother.update_person(obj_id, is_drowsy, is_phone)
            
            # Set final status and score based on smoothed results
            if smoothed_status == "USING PHONE":
                status, score = "USING PHONE", 40
            elif smoothed_status == "DROWSY":
                status, score = "DROWSY", 20
            else:
                status, score = "ATTENTIVE", 90

        result["persons"].append({
            "id": int(obj_id),
            "status": status,
            "score": score,
            "bbox": bbox
        })
        scores.append(score)

    if scores:
        result["overall_score"] = float(np.mean(scores))

    return result


# ADD GEMINI AI HELPER FUNCTION
def create_session_analysis_prompt(session_data):
    """Create a simple, safe prompt for Gemini AI analysis"""
    
    # Extract values one by one, safely
    class_name = "Unknown"
    duration_minutes = 0
    max_students = 0
    avg_attention = 0.0
    
    try:
        class_name = str(session_data.get("className", "Unknown"))
    except:
        pass
    
    try:
        duration_raw = session_data.get("duration", 0)
        duration_minutes = int(duration_raw / 60) if duration_raw else 0
    except:
        pass
        
    try:
        max_students = int(session_data.get("maxStudentsDetected", 0))
    except:
        pass
    
    # UPDATED: Handle your actual data structure
    try:
        scores = session_data.get("overallScores", [])
        print(f"DEBUG: Found overallScores with {len(scores)} items")
        
        if scores and len(scores) > 0:
            numeric_scores = []
            for score_obj in scores:
                if isinstance(score_obj, dict) and 'score' in score_obj:
                    # Extract score from {time: 2, score: 90, timestamp: 1758780003255}
                    numeric_scores.append(float(score_obj['score']))
                elif isinstance(score_obj, (int, float)):
                    # Handle plain numbers
                    numeric_scores.append(float(score_obj))
            
            if numeric_scores:
                avg_attention = sum(numeric_scores) / len(numeric_scores)
                print(f"DEBUG: Calculated average from {len(numeric_scores)} scores: {avg_attention}")
        
    except Exception as e:
        print(f"DEBUG: Error processing overallScores: {e}")
        avg_attention = 0.0
    
    # UPDATED PROMPT - Better formatting instructions
    prompt = f"""As an educational technology expert, analyze this classroom monitoring session and provide a comprehensive report.

**SESSION DATA:**
Class: {class_name}
Duration: {duration_minutes} minutes
Students Detected: {max_students}
Average Attention Score: {avg_attention:.1f}%

**ANALYSIS REQUIREMENTS:**
Provide a detailed educational assessment with clear, professional formatting. Even if the session is short, extract meaningful insights about classroom dynamics.

**FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:**

**1. Overall Session Quality**
Rate the session as Excellent/Good/Fair/Poor and provide a clear 2-sentence explanation.

**2. Key Educational Observations**
• [First specific observation about student engagement patterns]
• [Second observation about classroom attention dynamics]

**3. Actionable Teaching Recommendations**
• [First concrete suggestion for improving classroom engagement]
• [Second recommendation for enhancing student attention]
• [Third strategy for optimizing learning outcomes]

**IMPORTANT FORMATTING RULES:**
- Use clear section headers with numbers
- Use bullet points (•) for lists
- Keep language professional and educational-focused
- Provide actionable insights for teachers
- Keep total response under 400 words
- Focus on practical classroom management advice

Generate your analysis now."""
    
    return prompt



def generate_mock_ai_summary(session_data):
    """Generate a mock AI summary when Gemini is unavailable"""
    
    # Calculate metrics safely
    overall_scores = session_data.get("overallScores", [])
    avg_attention = float(np.mean(overall_scores)) if overall_scores else 0.0
    duration_seconds = session_data.get("duration", 0)
    duration_minutes = int(duration_seconds // 60) if duration_seconds else 0
    max_students = session_data.get("maxStudentsDetected", 0)
    class_name = str(session_data.get("className", "Unknown Class"))
    
    # Determine quality and insights based on attention score
    if avg_attention >= 85:
        quality = "Excellent"
        observations = [
            "High sustained attention throughout the session",
            "Strong student engagement with minimal distractions",
            "Optimal learning environment maintained"
        ]
        recommendations = [
            "Continue current teaching methods - they're highly effective",
            "Consider introducing advanced concepts during peak engagement periods",
            "Document successful strategies for replication in future sessions"
        ]
    elif avg_attention >= 75:
        quality = "Good"
        observations = [
            "Generally attentive class with some natural attention fluctuations",
            "Students maintained good focus for most of the session duration",
            "Minor distractions did not significantly impact overall engagement"
        ]
        recommendations = [
            "Implement brief interactive breaks during attention dip periods",
            "Use varied teaching methods to maintain consistent engagement",
            "Monitor and address minor distractions proactively"
        ]
    elif avg_attention >= 60:
        quality = "Fair"
        observations = [
            "Moderate attention levels with noticeable engagement fluctuations",
            "Several periods of decreased student focus observed",
            "Students showed signs of distraction or learning fatigue"
        ]
        recommendations = [
            "Increase interactive elements and hands-on activities",
            "Consider shorter lesson segments with strategic breaks",
            "Review classroom environment factors affecting attention",
            "Implement attention-checking techniques every 15 minutes"
        ]
    else:
        quality = "Needs Improvement"
        observations = [
            "Low attention levels throughout most of the session",
            "Frequent distractions and student disengagement patterns",
            "Students struggled to maintain focus on lesson content"
        ]
        recommendations = [
            "Redesign lesson structure with more interactive components",
            "Implement attention-getting strategies every 10-15 minutes",
            "Address underlying engagement issues through varied activities",
            "Consider smaller group work to increase individual participation",
            "Review and adjust teaching pace and content complexity"
        ]

    # Create observations text
    observations_text = ""
    for obs in observations:
        observations_text += f"• {obs}\n"
    
    # Create recommendations text
    recommendations_text = ""
    for rec in recommendations:
        recommendations_text += f"• {rec}\n"

    # Format the response as a single string (NO dict operations)
    mock_summary = f"""**Overall Session Quality: {quality}**

The {class_name} session ({duration_minutes} minutes, {max_students} students) achieved an average attention score of {avg_attention:.1f}%.

**Key Observations:**
{observations_text}
**Recommendations:**
{recommendations_text}
**Analysis Note:** This comprehensive analysis was generated using EduGauge's built-in intelligence system, providing data-driven insights for classroom improvement."""
    
    return mock_summary


# Connect to MongoDB on startup
@app.on_event("startup")
async def startup_event():
    await db_state.connect()

# API Endpoints
@app.get("/")
async def root():
    db_status = "connected" if db_state.connected else "not available"
    ai_status = "available" if gemini_model else "not configured"
    return {
        "message": "EduGauge WebSocket Backend with MongoDB and Gemini AI",
        "version": "1.1.0",
        "database": db_status,
        "ai_analysis": ai_status,
        "ai_model": working_model_name if working_model_name else "none",
        "endpoints": {
            "websocket": "/ws",
            "save_session": "POST /api/sessions",
            "get_sessions": "GET /api/sessions",
            "ai_summary": "POST /api/ai-summary",
            "quick_insights": "POST /api/quick-ai-insights",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    if db_state.connected and db_state.db is not None:
        try:
            await db_state.db.command("ping")
            return {
                "status": "healthy", 
                "database": "connected",
                "ai_analysis": "available" if gemini_model else "not configured",
                "ai_model": working_model_name if working_model_name else "none"
            }
        except Exception as e:
            return {
                "status": "healthy", 
                "database": f"connection error: {str(e)}",
                "ai_analysis": "available" if gemini_model else "not configured"
            }
    else:
        return {
            "status": "healthy", 
            "database": "not available",
            "ai_analysis": "available" if gemini_model else "not configured"
        }

# Session management endpoints (unchanged)
@app.post("/api/sessions")
async def save_session(session: dict = Body(...)):
    if not db_state.connected or db_state.db is None:
        return {"success": False, "message": "Database not available - data not saved"}
    
    try:
        if "createdAt" not in session:
            session["createdAt"] = datetime.utcnow()
        
        result = await db_state.db.sessions.insert_one(session)
        return {
            "success": True,
            "message": "Session saved successfully",
            "session_id": str(result.inserted_id)
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to save session: {str(e)}"}

@app.get("/api/sessions")
async def get_sessions():
    if not db_state.connected or db_state.db is None:
        return {"success": False, "sessions": [], "message": "Database not available"}
    
    try:
        cursor = db_state.db.sessions.find().sort("createdAt", -1).limit(100)
        sessions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            sessions.append(doc)
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        return {"success": False, "sessions": [], "message": f"Failed to retrieve sessions: {str(e)}"}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if not db_state.connected or db_state.db is None:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        result = await db_state.db.sessions.delete_one({"id": session_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"success": True, "message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

# IMPROVED GEMINI AI ENDPOINTS WITH RATE LIMITING
@app.post("/api/ai-summary")
async def generate_ai_summary(session_data: dict = Body(...)):
    """Generate AI summary for session data with robust error handling"""
    global last_ai_request
    
    # Try Gemini AI first
    if gemini_model:
        # Rate limiting
        time_since_last = datetime.now() - last_ai_request
        if time_since_last.total_seconds() < AI_REQUEST_DELAY:
            wait_time = AI_REQUEST_DELAY - time_since_last.total_seconds()
            await asyncio.sleep(wait_time)
        
        try:
            # Create analysis prompt
            analysis_prompt = create_session_analysis_prompt(session_data)
            
            # Generate AI analysis
            response = gemini_model.generate_content(analysis_prompt)
            ai_summary = response.text
            
            last_ai_request = datetime.now()
            
            # Save AI analysis to database if available
            if db_state.connected and db_state.db is not None:
                try:
                    ai_record = {
                        "session_id": session_data.get("id", "unknown"),
                        "session_name": session_data.get("className", "Unknown"),
                        "ai_summary": ai_summary,
                        "analysis_timestamp": datetime.utcnow(),
                        "ai_model": working_model_name,
                        "analysis_type": "gemini_ai",
                        "session_metrics": {
                            "avg_attention": np.mean(session_data.get("overallScores", [0])) if session_data.get("overallScores") else 0,
                            "duration_minutes": (session_data.get("duration", 0)) // 60,
                            "max_students": session_data.get("maxStudentsDetected", 0),
                            "total_distractions": len(session_data.get("distractionEvents", []))
                        }
                    }
                    await db_state.db.ai_analyses.insert_one(ai_record)
                except Exception as db_error:
                    print(f"Failed to save AI analysis to database: {db_error}")
            
            return {
                "success": True,
                "ai_summary": ai_summary,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "session_analyzed": session_data.get("className", "Unknown"),
                "ai_model_used": working_model_name,
                "analysis_type": "gemini_ai"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"Gemini AI failed: {error_msg[:100]}")
            
            # Fall back to mock AI if any error occurs
            print("🔄 Falling back to mock AI due to error...")
    
    # Use mock AI (either because Gemini unavailable or failed)
    try:
        mock_summary = generate_mock_ai_summary(session_data)
        
        # Save mock analysis to database if available
        if db_state.connected and db_state.db is not None:
            try:
                ai_record = {
                    "session_id": session_data.get("id", "unknown"),
                    "session_name": session_data.get("className", "Unknown"),
                    "ai_summary": mock_summary,
                    "analysis_timestamp": datetime.utcnow(),
                    "ai_model": "EduGauge Intelligence System",
                    "analysis_type": "mock_ai",
                    "session_metrics": {
                        "avg_attention": np.mean(session_data.get("overallScores", [0])) if session_data.get("overallScores") else 0,
                        "duration_minutes": (session_data.get("duration", 0)) // 60,
                        "max_students": session_data.get("maxStudentsDetected", 0),
                        "total_distractions": len(session_data.get("distractionEvents", []))
                    }
                }
                await db_state.db.ai_analyses.insert_one(ai_record)
            except Exception as db_error:
                print(f"Failed to save mock AI analysis: {db_error}")
        
        return {
            "success": True,
            "ai_summary": mock_summary,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "session_analyzed": session_data.get("className", "Unknown"),
            "ai_model_used": "EduGauge Intelligence System",
            "analysis_type": "mock_ai",
            "note": "Generated using EduGauge's built-in analysis while AI services are temporarily limited"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Analysis failed: {str(e)}"
        }


@app.post("/api/quick-ai-insights")
async def quick_ai_insights(session_data: dict = Body(...)):
    """Generate quick AI insights with fallback"""
    
    # Try Gemini first, fall back to mock if quota exceeded
    if gemini_model:
        try:
            avg_score = np.mean(session_data.get("overallScores", [0])) if session_data.get("overallScores") else 0
            duration = (session_data.get("duration", 0)) // 60
            
            quick_prompt = f"""
{session_data.get("className", "Class")}: {duration}min, {avg_score:.0f}% attention, {session_data.get("maxStudentsDetected", 0)} students.

Provide 3 brief points:
• Performance:
• Key observation:
• Recommendation:

Keep each under 15 words.
"""
            
            response = gemini_model.generate_content(quick_prompt)
            
            return {
                "success": True,
                "quick_insights": response.text,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_type": "gemini_ai"
            }
            
        except Exception as e:
            if "quota" in str(e).lower():
                print("🔄 Using mock insights due to quota limits...")
            
    # Mock quick insights
    avg_score = np.mean(session_data.get("overallScores", [0])) if session_data.get("overallScores") else 0
    
    if avg_score >= 85:
        insights = "• Performance: Excellent engagement with high attention levels\n• Key observation: Students maintained consistent focus throughout\n• Recommendation: Continue current teaching methods and increase complexity"
    elif avg_score >= 75:
        insights = "• Performance: Good attention with minor fluctuations\n• Key observation: Generally engaged with some distraction periods\n• Recommendation: Add interactive breaks during attention dips"
    elif avg_score >= 60:
        insights = "• Performance: Fair engagement with noticeable attention gaps\n• Key observation: Students showed signs of fatigue mid-session\n• Recommendation: Implement shorter segments with variety activities"
    else:
        insights = "• Performance: Low attention requiring immediate intervention\n• Key observation: Frequent distractions and disengagement\n• Recommendation: Redesign lesson structure with interactive components"
    
    return {
        "success": True,
        "quick_insights": insights,
        "timestamp": datetime.utcnow().isoformat(),
        "analysis_type": "mock_ai",
        "note": "Generated using built-in analysis system"
    }


# WebSocket endpoint (unchanged)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔗 WebSocket connection established")
    
    while True:
        try:
            data = await ws.receive_bytes()
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            result = analyze_frame(frame)
            await ws.send_json(result)
        except Exception as e:
            print("WebSocket error:", e)
            break



if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting EduGauge with MongoDB integration and Gemini AI...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
