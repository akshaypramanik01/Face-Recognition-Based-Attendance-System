# app.py - OPTIMIZED VERSION
import os
import time
import logging
import threading
from flask import Flask
from flask_cors import CORS
from pymongo import MongoClient
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
import numpy as np

# Blueprint imports
from auth.routes import auth_bp

# Optional student/teacher blueprints
try:
    from student.registration import student_registration_bp
except ImportError:
    student_registration_bp = None

try:
    from student.updatedetails import student_update_bp
except ImportError:
    student_update_bp = None

try:
    from student.demo_session import demo_session_bp
except ImportError:
    demo_session_bp = None

try:
    from student.view_attendance import attendance_bp
except ImportError:
    attendance_bp = None

try:
    from teacher.attendance_records import attendance_session_bp
except ImportError:
    attendance_session_bp = None

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DATABASE_NAME", "facerecognition")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "students")
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
students_collection = db[COLLECTION_NAME]
attendance_db = client["facerecognition_db"]
attendance_collection = attendance_db["attendance_records"]


# OPTIMIZED MODEL MANAGER CLASS
class ModelManager:
    """
    Lazy / background model loader.
    - Constructing the singleton is cheap and non-blocking.
    - Models load in background thread.
    - Exposes .is_ready() and .wait_until_ready(timeout) helpers.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # set defaults quickly
                    cls._instance.models_ready = False
                    cls._instance._loading = False
                    cls._instance.detector = None
                    cls._instance.deepface_ready = False
                    cls._instance._init_thread = None
        return cls._instance

    def start_background_initialization(self, force_restart=False):
        """Start (or restart) the background initialization thread if not already running."""
        if self._loading and not force_restart:
            logger.info("ModelManager: initialization already in progress")
            return
        self._init_thread = threading.Thread(target=self._initialize_models_safe, daemon=True)
        self._init_thread.start()

    def _initialize_models_safe(self):
        """Wrapper to initialize models and catch exceptions (runs in background thread)."""
        self._loading = True
        try:
            self._initialize_models()
        except Exception as e:
            logger.exception("ModelManager: background initialization failed: %s", e)
            self.models_ready = False
            self.deepface_ready = False
        finally:
            self._loading = False

    def _initialize_models(self):
        """Actual initialization (heavy). Run in background thread."""
        logger.info("ModelManager: Starting model initialization in background thread...")
        start_time = time.time()
        # Reset flags
        self.models_ready = False
        self.deepface_ready = False
        self.detector = None

        # --- 1) Initialize MTCNN detector ---
        try:
            from mtcnn import MTCNN
            logger.info("ModelManager: Loading MTCNN detector...")
            # Optionally tune detector parameters to trade accuracy for speed if needed
            self.detector = MTCNN()
            logger.info("ModelManager: MTCNN loaded")
        except Exception as e:
            logger.exception("ModelManager: failed to load MTCNN: %s", e)
            raise

        # --- 2) Initialize DeepFace model with minimal warmup ---
        try:
            # Import locally to avoid heavy import at module load time
            from deepface import DeepFace
            logger.info("ModelManager: Loading DeepFace (Facenet512)...")

            # Perform a single, minimal warmup using enforce_detection=False
            # Use a tiny grayscale image scaled to required size to avoid heavy memory copies
            dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)

            # Only call represent once - this forces model build/download
            _ = DeepFace.represent(
                dummy_img,
                model_name='Facenet512',
                detector_backend='skip',
                enforce_detection=False
            )

            self.deepface_ready = True
            logger.info("ModelManager: DeepFace warmed up")
        except Exception as e:
            logger.exception("ModelManager: failed to init DeepFace: %s", e)
            # do not re-raise - background initialization should fail gracefully
            self.deepface_ready = False

        # Mark ready if both components are fine
        self.models_ready = (self.detector is not None) and self.deepface_ready
        elapsed = time.time() - start_time
        logger.info("ModelManager: initialization finished (ready=%s) in %.2fs", self.models_ready, elapsed)

    def is_ready(self):
        """Return True only if models are fully loaded and warmed up."""
        return bool(self.models_ready)

    def wait_until_ready(self, timeout=None):
        """Blocking wait for initialization to finish (optional). Returns True if ready."""
        t0 = time.time()
        while True:
            if self.is_ready():
                return True
            # if initialization thread finished but not ready -> return False
            if self._init_thread and not self._init_thread.is_alive():
                return self.is_ready()
            if timeout is not None and (time.time() - t0) > timeout:
                return False
            time.sleep(0.2)

    def get_detector(self):
        """Return detector if ready else raise. Callers should guard by is_ready()."""
        if not self.is_ready():
            raise RuntimeError("Models not yet ready")
        return self.detector

    def health_check(self, light=True):
        """
        Light health check:
         - If light=True: return basic flags without any heavy inference.
         - If light=False: run a light inference (may be slow).
        """
        if not self._loading and not self.is_ready():
            # not loading and not ready -> failed earlier
            return False

        if light:
            # No heavy inference - rely on ready flags
            return self.is_ready()
        else:
            # heavier check (optional)
            try:
                # quick detector call with a small random image
                test_img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
                _ = self.detector.detect_faces(test_img)
                from deepface import DeepFace
                test_face = np.zeros((160, 160, 3), dtype=np.uint8)
                _ = DeepFace.represent(test_face, model_name='Facenet512', detector_backend='skip',
                                       enforce_detection=False)
                return True
            except Exception as e:
                logger.exception("ModelManager: health check failed: %s", e)
                return False


# Flask app
app = Flask(__name__)
CORS(app)

# Initialize the model manager (singleton)
logger.info("Initializing Model Manager...")
# create the singleton (cheap)
model_manager = ModelManager()
# start loading models in background
model_manager.start_background_initialization()
# put into app.config so blueprints can access it
app.config["MODEL_MANAGER"] = model_manager

# Configure Flask app with database and model instances
app.config["DB"] = db
app.config["COLLECTION_NAME"] = COLLECTION_NAME
app.config["THRESHOLD"] = THRESHOLD
app.config["ATTENDANCE_COLLECTION"] = attendance_collection

# CRITICAL: Pass model manager to Flask config so blueprints can access it
app.config["MODEL_MANAGER"] = model_manager

bcrypt = Bcrypt(app)


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    mm = app.config.get("MODEL_MANAGER")
    if mm is None:
        return {"status": "no-model-manager"}, 500

    if mm.is_ready():
        # do a light health check (non-heavy)
        return {"status": "healthy", "models_ready": True, "models_healthy": True, "timestamp": time.time()}
    else:
        # either still loading or failed
        loading = getattr(mm, "_loading", False)
        return {
            "status": "initializing" if loading else "unhealthy",
            "models_ready": mm.is_ready(),
            "loading": loading,
            "timestamp": time.time()
        }


# Register blueprints
app.register_blueprint(auth_bp)

if student_registration_bp:
    app.register_blueprint(student_registration_bp)
    logger.info("✅ Student registration blueprint registered")

if student_update_bp:
    app.register_blueprint(student_update_bp)
    logger.info("✅ Student update blueprint registered")

if demo_session_bp:
    app.register_blueprint(demo_session_bp)
    logger.info("✅ Demo session blueprint registered")

if attendance_bp:
    app.register_blueprint(attendance_bp)
    logger.info("✅ Attendance blueprint registered")

if attendance_session_bp:
    app.register_blueprint(attendance_session_bp)
    logger.info("✅ Attendance session blueprint registered")

# List all registered routes
logger.info("\nRegistered Flask Routes:")
for rule in app.url_map.iter_rules():
    logger.info(f"  {rule}")

if __name__ == "__main__":
    logger.info("🚀 Starting Flask server...")

    # Start server - models will load in background
    if model_manager.is_ready():
        logger.info("🎯 All systems ready! Server starting on http://0.0.0.0:5000")
    else:
        logger.info("⏳ Server starting while models load in background... Check /health for status")

    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False for production