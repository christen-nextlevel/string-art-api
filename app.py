# app.py
import os, uuid, tempfile, hashlib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict
import requests

from pipeline.string_art_matplotlib import generate_string_art
from pipeline.string_art_timelapse import make_timelapse
from pipeline.csv_to_pdf_nopandas import csv_to_pdf

# ====== CONFIG ======
WIX_WEBHOOK_DONE = os.getenv("WIX_WEBHOOK_DONE")  # e.g., https://your-site/_functions/markJobDone
STORAGE_BUCKET   = os.getenv("STORAGE_BUCKET")    # if using S3
USE_S3           = bool(STORAGE_BUCKET)
AWS_REGION       = os.getenv("AWS_REGION", "eu-west-1")
# ====================

app = FastAPI(title="String Art API")

# naive in-memory job store (swap for Redis/DB in production)
JOBS: Dict[str, Dict] = {}

# ---- helpers ----
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def download_to_tmp(url: str) -> str:
    r = requests.get(url, timeout=60)
    if not r.ok:
        raise HTTPException(status_code=400, detail="Unable to download image")
    fd, p = tempfile.mkstemp(suffix=Path(url).suffix or ".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return p

def upload_file(local_path: str) -> str:
    """
    Return a URL. For MVP you can use local disk or a simple static host.
    If using S3:
    """
    if not USE_S3:
        # DEV: serve from e.g. Railway persistent volume + static (or return file for testing)
        return f"file://{local_path}"

    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    key = f"string-art/{uuid.uuid4().hex}/{Path(local_path).name}"
    s3.upload_file(local_path, STORAGE_BUCKET, key, ExtraArgs={'ContentType': guess_mime(local_path), 'ACL': 'public-read'})
    return f"https://{STORAGE_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

def guess_mime(p: str) -> str:
    ext = Path(p).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg":"image/jpeg",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".txt": "text/plain",
    }.get(ext, "application/octet-stream")

# ---- models ----
class RedeemIn(BaseModel):
    code: str
    imageUrl: HttpUrl
    # optional settings you may expose later
    numNails: Optional[int] = 240
    numLines: Optional[int] = 1300

class RedeemOut(BaseModel):
    jobId: str

class StatusOut(BaseModel):
    status: str
    progress: int = 0
    timelapseUrl: Optional[str] = None
    pdfUrl: Optional[str] = None
    csvUrl: Optional[str] = None
    resultImageUrl: Optional[str] = None
    error: Optional[str] = None

# ---- routes ----
@app.post("/redeem", response_model=RedeemOut)
def redeem(inp: RedeemIn):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "progress": 0}

    # Immediately spawn a background-like task (simple thread for MVP)
    import threading
    t = threading.Thread(target=_process_job, args=(job_id, inp,), daemon=True)
    t.start()

    return RedeemOut(jobId=job_id)

@app.get("/status/{job_id}", response_model=StatusOut)
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return StatusOut(**job)

# ---- core pipeline ----
def _process_job(job_id: str, inp: RedeemIn):
    try:
        JOBS[job_id].update(status="processing", progress=5)
        # (1) download image
        img_local = download_to_tmp(str(inp.imageUrl))

        # Work dir
        work = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_"))
        # (2) string art + CSV + TXT
        JOBS[job_id]["progress"] = 30
        sa = generate_string_art(
            image_path=img_local,
            out_dir=str(work / "string_art"),
            num_nails=inp.numNails,
            num_lines=inp.numLines,
            lighten_to=1.0
        )

        # (3) timelapse (re-run drawing to produce frames/video)
        JOBS[job_id]["progress"] = 60
        tl = make_timelapse(
            image_path=img_local,
            out_dir=str(work / "timelapse"),
            num_nails=inp.numNails,
            num_lines=inp.numLines,
            lighten_to=1.0,
            snapshot_every=25,
            mp4=True, gif=False
        )

        # (4) CSV -> PDF
        JOBS[job_id]["progress"] = 80
        pdf_path = str(work / "string_art_instructions.pdf")
        csv_to_pdf(sa["lines_csv"], pdf_path)

        # (5) upload artifacts
        result_img_url = upload_file(sa["result_png"])
        csv_url       = upload_file(sa["lines_csv"])
        pdf_url       = upload_file(pdf_path)
        tl_url        = upload_file(tl.get("timelapse_mp4") or tl.get("timelapse_gif") or tl["final_png"])

        # (6) done
        JOBS[job_id].update(
            status="done",
            progress=100,
            resultImageUrl=result_img_url,
            csvUrl=csv_url,
            pdfUrl=pdf_url,
            timelapseUrl=tl_url,
        )

        # (7) notify Wix (optional but recommended)
        if WIX_WEBHOOK_DONE:
            try:
                requests.post(WIX_WEBHOOK_DONE, json={
                    "jobId": job_id,
                    "status": "done",
                    "timelapseUrl": tl_url,
                    "pdfUrl": pdf_url,
                    "csvUrl": csv_url,
                    "resultImageUrl": result_img_url
                }, timeout=20)
            except Exception:
                pass

    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e), progress=100)
        if WIX_WEBHOOK_DONE:
            try:
                requests.post(WIX_WEBHOOK_DONE, json={
                    "jobId": job_id, "status": "error", "error": str(e)
                }, timeout=20)
            except Exception:
                pass