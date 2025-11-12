import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import logging
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from lightcurve_build_pipeline import run_pipeline
from config import FastPipelineConfig
from inference import load_model, run_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and temporary storage
model_session = None
temp_dir = Path("./api_temp")
temp_dir.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global model_session
    try:
        model_path = os.getenv("MODEL_PATH", "models/koi_rf_model.onnx")
        if Path(model_path).exists():
            model_session = load_model(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Lightcurve Analysis API",
    description="API for processing lightcurve FITS files through the complete analysis pipeline",
    version="1.0.0",
    lifespan=lifespan
)

default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

_allowed_origins_raw = os.getenv("LIGHTCURVE_ALLOWED_ORIGINS")
if _allowed_origins_raw:
    allowed_origins = [origin.strip() for origin in _allowed_origins_raw.split(",") if origin.strip()]
else:
    allowed_origins = default_origins

allow_credentials = True
if allowed_origins == ["*"]:
    allow_credentials = False

# Enable CORS for frontend access (configurable via env var LIGHTCURVE_ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

def save_lightcurve_as_csv(lightcurve, file_path: Path) -> Path:
    """Save a lightcurve object as CSV file"""
    try:
        # Extract time/flux arrays without relying on pandas to reduce memory footprint
        time_attr = getattr(lightcurve, "time", None)
        flux_attr = getattr(lightcurve, "flux", None)

        if time_attr is None or flux_attr is None:
            raise ValueError("Lightcurve object missing time or flux data")

        if hasattr(time_attr, "to_value"):
            time_array = np.asarray(time_attr.to_value(), dtype=np.float64)
        elif hasattr(time_attr, "value"):
            time_array = np.asarray(time_attr.value, dtype=np.float64)
        else:
            time_array = np.asarray(time_attr, dtype=np.float64)

        if hasattr(flux_attr, "value"):
            flux_array = np.asarray(flux_attr.value, dtype=np.float64)
        else:
            flux_array = np.asarray(flux_attr, dtype=np.float64)

        mask = np.isfinite(time_array) & np.isfinite(flux_array)
        filtered_time = time_array[mask]
        filtered_flux = flux_array[mask]

        if filtered_time.size == 0 or filtered_flux.size == 0:
            raise ValueError("Lightcurve contains no finite values for time/flux")

        stacked = np.column_stack((filtered_time, filtered_flux))
        np.savetxt(file_path, stacked, delimiter=",", header="time,flux", comments="")
        return file_path

    except Exception as e:
        logger.error(f"Error saving lightcurve as CSV: {e}")
        raise

@app.post("/api/v1/upload-fits")
async def upload_fits_file(file: UploadFile = File(...)):
    """
    Upload a FITS file and process it through the lightcurve pipeline.
    Returns a task ID for tracking the processing status.
    """
    if not file.filename.endswith('.fits'):
        raise HTTPException(status_code=400, detail="File must be a FITS file (.fits extension)")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    task_dir = temp_dir / task_id
    task_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded file without loading entire payload into memory
        fits_path = task_dir / file.filename
        chunk_size = 1024 * 1024  # 1 MB chunks to keep memory usage low
        with open(fits_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        await file.close()
        
        logger.info(f"Saved FITS file for task {task_id}: {fits_path}")
        
        # Run the pipeline with fast configuration for web usage
        cfg = FastPipelineConfig()
        logger.info("Using FastPipelineConfig for optimized web performance")
        result = run_pipeline(local_paths=str(fits_path), cfg=cfg)
        
        # Generate plots (lazy import keeps heavy deps out of cold start path)
        plot_dir = task_dir / "plots"
        from plotting import plot_results  # Lazy import keeps dependency optional
        plot_results(result, outdir=plot_dir)
        
        # Save lightcurve data for later use
        lc_csv_path = task_dir / "lightcurve.csv"
        save_lightcurve_as_csv(result.folded_lc, lc_csv_path)
        
        return JSONResponse({
            "task_id": task_id,
            "status": "completed",
            "message": "FITS file processed successfully",
            "best_period": result.best_period
        })
        
    except Exception as e:
        logger.error(f"Error processing FITS file for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing FITS file: {str(e)}")

@app.get("/api/v1/folded-image/{task_id}")
async def get_folded_image(task_id: str):
    """Get the folded lightcurve image for a given task ID"""
    task_dir = temp_dir / task_id
    image_path = task_dir / "plots" / "03_folded.png"
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Folded image not found")
    
    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"folded_lightcurve_{task_id}.png"
    )

@app.post("/api/v1/transit-parameters/{task_id}")
async def get_transit_parameters(task_id: str):
    """
    Extract transit parameters using the fast_transit workflow.
    Returns 4 parameters: t0, depth, duration, baseline (excluding ingress_fit).
    """
    task_dir = temp_dir / task_id
    lc_csv_path = task_dir / "lightcurve.csv"
    
    if not lc_csv_path.exists():
        raise HTTPException(status_code=404, detail="Lightcurve data not found for this task ID")
    
    try:
        # Read the lightcurve CSV with NumPy to avoid pandas overhead
        data = np.genfromtxt(lc_csv_path, delimiter=",", names=True, dtype=np.float64)
        if data.size == 0:
            raise ValueError("Lightcurve CSV is empty")

        time = np.atleast_1d(data["time"])
        flux = np.atleast_1d(data["flux"])

        finite_mask = np.isfinite(time) & np.isfinite(flux)
        if not np.any(finite_mask):
            raise ValueError("No finite values found in lightcurve data")

        time = time[finite_mask]
        flux = flux[finite_mask]
        
        # Fit trapezoid model
        from fast_transit import fit_trapezoid_from_arrays
        popt, _ = fit_trapezoid_from_arrays(time, flux, outdir=task_dir / "transit_plots")
        
        # Extract 4 parameters (excluding ingress_fit which is at index 3)
        t0_fit, depth_fit, duration_fit, ingress_fit, baseline_fit = popt
        
        # Return 4 parameters as requested
        parameters = {
            "t0": float(t0_fit),
            "depth": float(depth_fit), 
            "duration": float(duration_fit),
            "baseline": float(baseline_fit)
        }
        
        # Store parameters for inference step
        params_file = task_dir / "transit_params.json"
        import json
        with open(params_file, 'w') as f:
            json.dump(parameters, f)
        
        return JSONResponse({
            "task_id": task_id,
            "status": "success",
            "parameters": parameters,
            "message": "Transit parameters extracted successfully"
        })
        
    except Exception as e:
        logger.error(f"Error extracting transit parameters for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting transit parameters: {str(e)}")

@app.post("/api/v1/ml-inference/{task_id}")
async def run_ml_inference(task_id: str):
    """
    Run ML model inference using the extracted transit parameters.
    """
    global model_session
    
    if model_session is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    task_dir = temp_dir / task_id
    params_file = task_dir / "transit_params.json"
    
    if not params_file.exists():
        raise HTTPException(
            status_code=404, 
            detail="Transit parameters not found. Please run transit parameter extraction first."
        )
    
    try:
        # Load transit parameters
        import json
        with open(params_file, 'r') as f:
            parameters = json.load(f)
        
        # Prepare input for ML model (4 features)
        input_features = np.array([[
            parameters["t0"],
            parameters["depth"], 
            parameters["duration"],
            parameters["baseline"]
        ]], dtype=np.float32)
        
        # Run inference
        results = run_inference(model_session, {"float_input": input_features})
        
        # Convert numpy arrays to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        
        return JSONResponse({
            "task_id": task_id,
            "status": "success",
            "input_parameters": parameters,
            "inference_results": serializable_results,
            "message": "ML inference completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error running ML inference for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error running ML inference: {str(e)}")

@app.get("/api/v1/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    task_dir = temp_dir / task_id
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    # Check what files exist to determine status
    fits_files = list(task_dir.glob("*.fits"))
    plots_exist = (task_dir / "plots" / "03_folded.png").exists()
    params_exist = (task_dir / "transit_params.json").exists()
    
    status = "unknown"
    if fits_files and plots_exist:
        status = "pipeline_completed"
    if params_exist:
        status = "parameters_extracted"
    
    return JSONResponse({
        "task_id": task_id,
        "status": status,
        "pipeline_completed": plots_exist,
        "parameters_extracted": params_exist
    })

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "model_loaded": model_session is not None,
        "version": "1.0.0"
    })

@app.delete("/api/v1/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    """Clean up temporary files for a task"""
    task_dir = temp_dir / task_id
    
    if task_dir.exists():
        import shutil
        shutil.rmtree(task_dir)
        return JSONResponse({"message": f"Task {task_id} cleaned up successfully"})
    else:
        raise HTTPException(status_code=404, detail="Task ID not found")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        log_level="info"
    )