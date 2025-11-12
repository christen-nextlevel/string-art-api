import os
import io
import uuid
import shutil
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from skimage import io as skio, color, draw

# Optional video
try:
    import imageio.v2 as imageio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# PDF export
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

# --------------------------------------------------------------------------------------
# Config / Paths
# --------------------------------------------------------------------------------------

app = FastAPI(title="String Art API")

FILES_ROOT = Path(os.getenv("FILES_ROOT", "jobs"))  # where we keep generated artifacts
FILES_ROOT.mkdir(parents=True, exist_ok=True)

# Serve files:  https://<BASE>/files/<jobId>/<file>
app.mount("/files", StaticFiles(directory=str(FILES_ROOT)), name="files")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")  # set on Render (e.g., https://string-art-api.onrender.com)

# In-memory job store (MVP/testing). For production, use a DB/redis.
_JOBS: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class RedeemPayload(BaseModel):
    imageUrl: str
    settings: Optional[Dict[str, Any]] = None
    # you can include "code" or other fields here if you want, they are ignored by the API for now
    code: Optional[str] = None


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _job_dir(job_id: str) -> Path:
    d = FILES_ROOT / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def _public_url(job_id: str, filename: str) -> str:
    rel = f"/files/{job_id}/{filename}"
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}{rel}"
    return rel  # relative (works for curl/local)

def _download_to(url: str, dest_path: Path):
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download image: HTTP {r.status_code}")
    dest_path.write_bytes(r.content)

def _save_frame(canvas: np.ndarray, out_path: Path, dpi: int = 200):
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

def _csv_to_pdf(csv_path: Path, pdf_path: Path, title="String Art Threading Instructions",
                subtitle="Nail 1 is at the top (12 o'clock), numbers increase clockwise.",
                landscape_mode: bool = False):
    """
    Minimal no-pandas CSV->PDF table (compatible with your script’s output).
    """
    import csv
    page_size = landscape(A4) if landscape_mode else A4
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=page_size,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    story = []
    story.append(Paragraph(f"<b>{title}</b>", _pdf_styles()["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(subtitle, _pdf_styles()["Normal"]))
    story.append(Spacer(1, 10))

    header = ["Step", "From Nail", "To Nail"]
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # tolerate header variants
        fn = None
        tn = None
        if reader.fieldnames:
            fns = {k.lower().strip(): k for k in reader.fieldnames}
            fn = fns.get("from_nail") or fns.get("from") or reader.fieldnames[0]
            tn = fns.get("to_nail") or fns.get("to") or reader.fieldnames[1]
        for i, row in enumerate(reader, start=1):
            rows.append([i, str(row[fn]).strip(), str(row[tn]).strip()])

    table = Table([header] + rows, colWidths=[25 * mm, 45 * mm, 45 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    story.append(table)
    doc.build(story)

_pdf_style_cache = None
def _pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet
    global _pdf_style_cache
    if _pdf_style_cache is None:
        _pdf_style_cache = getSampleStyleSheet()
    return _pdf_style_cache


# --------------------------------------------------------------------------------------
# Core String Art Routine (single pass does image + csv + frames)
# --------------------------------------------------------------------------------------

def run_string_art_with_frames(
    input_image_path: Path,
    out_dir: Path,
    num_nails: int = 240,
    num_lines: int = 1300,
    lighten_to: float = 1.0,
    snapshot_every: int = 25,
    make_gif: bool = False,
    gif_fps: int = 20,
    make_mp4: bool = True,
    mp4_fps: int = 30,
) -> Dict[str, Any]:
    """
    Returns dict with keys:
      result_png, lines_csv, instr_txt, frames_dir, timelapse_gif (opt), timelapse_mp4 (opt)
    """
    # --- load & grayscale (RGB/RGBA/gray) ---
    img = skio.imread(str(input_image_path))
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = color.rgba2rgb(img.astype(np.float32) / 255.0)
        if img.max() > 1.0:
            img = img / 255.0
        img = color.rgb2gray(img)
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

    # --- square crop (top-left) ---
    h, w = img.shape
    r = min(h // 2, w // 2)
    img = img[0:2 * r, 0:2 * r]
    L = img.shape[0]

    # --- nails on circle ---
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    rows = (L / 2 + (L * 0.5) * np.cos(angles)).astype(int)
    cols = (L / 2 + (L * 0.5) * np.sin(angles)).astype(int)
    nails = list(zip(rows, cols))  # (row, col)

    # --- canvas (white) ---
    canvas = np.ones_like(img)

    # outputs
    result_png = out_dir / "string_art_result.png"
    lines_csv = out_dir / "string_art_lines.csv"
    instr_txt = out_dir / "string_art_instructions.txt"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # save initial blank frame
    if snapshot_every and snapshot_every > 0:
        _save_frame(canvas, frames_dir / f"frame_{0:05d}.png")

    # drawing loop
    current = 0
    order = [current + 1]
    lines = []

    for i in range(num_lines):
        best_j = current
        best_avg = np.inf
        best_rr = best_cc = None

        cr, cc = nails[current]
        for j in range(num_nails):
            rr, cc2 = draw.line(cr, cc, nails[j][0], nails[j][1])
            rr = np.clip(rr, 0, L - 1)
            cc2 = np.clip(cc2, 0, L - 1)
            avg = float(np.mean(img[rr, cc2]))  # 0=dark, 1=light
            if avg < best_avg:
                best_avg = avg
                best_j = j
                best_rr, best_cc = rr, cc2

        if best_rr is not None:
            img[best_rr, best_cc] = lighten_to
            canvas[best_rr, best_cc] = 0.0

        lines.append((current + 1, best_j + 1))
        current = best_j
        order.append(current + 1)

        if snapshot_every and ((i + 1) % snapshot_every == 0):
            _save_frame(canvas, frames_dir / f"frame_{(i + 1):05d}.png")

    # final still image
    _save_frame(canvas, result_png, dpi=300)

    # write CSV
    with lines_csv.open("w", encoding="utf-8") as f:
        f.write("from_nail,to_nail\n")
        for a, b in lines:
            f.write(f"{a},{b}\n")

    # write one-line instructions
    with instr_txt.open("w", encoding="utf-8") as f:
        f.write("STRING ART THREAD ORDER (1-indexed, nail 1 at the top by convention):\n")
        for i in range(0, len(order), 25):
            f.write(" ".join(str(x) for x in order[i:i+25]) + "\n")

    # build GIF/MP4 from frames
    timelapse_gif = None
    timelapse_mp4 = None

    frame_files = sorted([p for p in frames_dir.glob("frame_*.png")])
    if HAVE_IMAGEIO and frame_files:
        if make_gif:
            imgs = [imageio.imread(str(p)) for p in frame_files]
            timelapse_gif = out_dir / "string_art_timelapse.gif"
            imageio.mimsave(str(timelapse_gif), imgs, fps=gif_fps)
        if make_mp4:
            try:
                import imageio_ffmpeg  # noqa: F401
                timelapse_mp4 = out_dir / "string_art_timelapse.mp4"
                writer = imageio.get_writer(
                    str(timelapse_mp4),
                    format="FFMPEG",
                    fps=mp4_fps,
                    codec="libx264",
                    quality=8,
                )
                for p in frame_files:
                    im = imageio.imread(str(p))
                    if im.ndim == 2:
                        im = np.stack([im, im, im], axis=-1)
                    if im.dtype != np.uint8:
                        im = np.clip(im, 0, 255).astype(np.uint8)
                    writer.append_data(im)
                writer.close()
            except Exception as e:
                # MP4 optional; ignore if ffmpeg not present on the platform
                print(f"[WARN] MP4 export failed: {e}")

    return {
        "result_png": str(result_png),
        "lines_csv": str(lines_csv),
        "instr_txt": str(instr_txt),
        "frames_dir": str(frames_dir),
        "timelapse_gif": str(timelapse_gif) if timelapse_gif else None,
        "timelapse_mp4": str(timelapse_mp4) if timelapse_mp4 else None,
    }


# --------------------------------------------------------------------------------------
# Job runner
# --------------------------------------------------------------------------------------

def _process_job(job_id: str, image_path: Path, settings: Dict[str, Any]):
    job = _JOBS[job_id]
    out_dir = _job_dir(job_id)

    # defaults (keep small for free tier testing)
    num_nails = int(settings.get("numNails", 240))
    num_lines = int(settings.get("numLines", 900))
    lighten_to = float(settings.get("lightenTo", 1.0))
    snapshot_every = int(settings.get("snapshotEvery", 25))
    make_gif = bool(settings.get("makeGif", False))
    make_mp4 = bool(settings.get("makeMp4", True))

    try:
        job["status"] = "processing"
        job["progress"] = 5

        # run the combined routine
        sa = run_string_art_with_frames(
            input_image_path=image_path,
            out_dir=out_dir,
            num_nails=num_nails,
            num_lines=num_lines,
            lighten_to=lighten_to,
            snapshot_every=snapshot_every,
            make_gif=make_gif,
            make_mp4=make_mp4,
        )
        job["progress"] = 80

        # build PDF from CSV
        csv_path = Path(sa["lines_csv"])
        pdf_path = out_dir / "string_art_instructions.pdf"
        _csv_to_pdf(csv_path, pdf_path)
        job["progress"] = 90

        # publish URLs
        job["resultImageUrl"] = _public_url(job_id, Path(sa["result_png"]).name)
        job["resultCsvUrl"] = _public_url(job_id, csv_path.name)
        job["resultPdfUrl"] = _public_url(job_id, pdf_path.name)

        # prefer MP4 if available, else GIF, else final PNG
        if sa.get("timelapse_mp4"):
            job["resultTimelapseUrl"] = _public_url(job_id, Path(sa["timelapse_mp4"]).name)
        elif sa.get("timelapse_gif"):
            job["resultTimelapseUrl"] = _public_url(job_id, Path(sa["timelapse_gif"]).name)
        else:
            job["resultTimelapseUrl"] = job["resultImageUrl"]

        job["status"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["progress"] = 100
        print(f"[ERROR] Job {job_id} failed: {e}")


# --------------------------------------------------------------------------------------
# API Routes
# --------------------------------------------------------------------------------------

@app.get("/")
def home():
    return {"status": "ok", "publicBaseUrl": PUBLIC_BASE_URL or "(relative)", "filesRoot": str(FILES_ROOT)}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    # don’t dump local paths to clients
    public = {k: v for k, v in job.items() if not k.endswith("_path")}
    return public

@app.post("/redeem")
def redeem(payload: RedeemPayload):
    """
    Starts a background job using an image URL (Wix upload URL is perfect).
    Returns a jobId you can poll with /status/{jobId}.
    """
    if not payload.imageUrl:
        raise HTTPException(status_code=400, detail="imageUrl required")

    # seed job
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "resultImageUrl": None,
        "resultPdfUrl": None,
        "resultCsvUrl": None,
        "resultTimelapseUrl": None,
    }

    # download image
    out_dir = _job_dir(job_id)
    input_path = out_dir / "input.jpg"
    try:
        _download_to(payload.imageUrl, input_path)
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = f"Download failed: {e}"
        return {"jobId": job_id, "status": "error", "error": str(e)}

    settings = payload.settings or {}

    # spawn background thread
    t = threading.Thread(target=_process_job, args=(job_id, input_path, settings), daemon=True)
    t.start()

    return {"jobId": job_id, "status": "queued"}

@app.post("/redeem-upload")
def redeem_upload(file: UploadFile = File(...)):
    """
    Alternative: send a file directly (multipart/form-data). Handy for local tests.
    """
    job_id = uuid.uuid4().hex[:12]
    _JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "resultImageUrl": None,
        "resultPdfUrl": None,
        "resultCsvUrl": None,
        "resultTimelapseUrl": None,
    }

    out_dir = _job_dir(job_id)
    input_path = out_dir / "input.jpg"
    input_path.write_bytes(file.file.read())

    settings = {}
    t = threading.Thread(target=_process_job, args=(job_id, input_path, settings), daemon=True)
    t.start()
    return {"jobId": job_id, "status": "queued"}