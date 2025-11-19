import os
import json
import uuid
from typing import Optional, Literal
import traceback

import numpy as np
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from starlette.responses import FileResponse
from concurrent.futures import ThreadPoolExecutor

import requests
from skimage import io, color, draw
import imageio.v2 as imageio

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
# =====================================================
# Ensure persistent jobs directory
# =====================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
# -------------------------------------------------------------------
# Basic config
# -------------------------------------------------------------------

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g. https://string-art-api.onrender.com
JOBS_ROOT = "jobs"
os.makedirs(JOBS_ROOT, exist_ok=True)

# Parameters for the string art generator
NUM_NAILS = 240
NUM_LINES = 100 #1300
LIGHTEN_TO = 1.0
SNAPSHOT_EVERY = 20 #25  # how often to snapshot for timelapse


app = FastAPI(title="String Art API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# A tiny thread pool so jobs run in the background
EXECUTOR = ThreadPoolExecutor(max_workers=2)


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------

class RedeemBody(BaseModel):
    imageUrl: HttpUrl


class JobStatus(BaseModel):
    jobId: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    resultImageUrl: Optional[str] = None
    resultPdfUrl: Optional[str] = None
    resultCsvUrl: Optional[str] = None
    resultTimelapseUrl: Optional[str] = None


# -------------------------------------------------------------------
# Helper functions for status JSON per job
# -------------------------------------------------------------------

def job_dir(job_id: str) -> str:
    return os.path.join(JOBS_ROOT, job_id)


def status_path(job_id: str) -> str:
    return os.path.join(job_dir(job_id), "status.json")


def read_status(job_id: str) -> JobStatus:
    path = status_path(job_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Unknown job_id")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JobStatus(**data)


def write_status(status: JobStatus) -> None:
    jd = job_dir(status.jobId)
    os.makedirs(jd, exist_ok=True)
    path = status_path(status.jobId)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status.dict(), f)


def build_file_url(job_id: str, filename: str) -> str:
    if PUBLIC_BASE_URL:
        base = PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/files/{job_id}/{filename}"
    # Fallback: relative path
    return f"/files/{job_id}/{filename}"


# -------------------------------------------------------------------
# Core pipeline: string art + timelapse + CSV → PDF
# -------------------------------------------------------------------

def generate_string_art_assets(input_path: str, job_id: str) -> None:
    """
    Runs the full pipeline for a given job:
    - generate string art PNG
    - generate frames + MP4 timelapse
    - generate CSV & PDF instructions
    Updates status.json as it goes.
    """
    jd = job_dir(job_id)
    os.makedirs(jd, exist_ok=True)

    status = JobStatus(jobId=job_id, status="processing")
    write_status(status)

    # 👇 DEBUG: log that the job started
    print(f"[JOB {job_id}] Starting pipeline, input_path={input_path}", flush=True)

    try:
        # ---------------------
        # Load & prepare image
        # ---------------------
        img = io.imread(input_path)

        if img.ndim == 3:
            if img.shape[2] == 4:  # RGBA
                img = color.rgba2rgb(img.astype(np.float32) / 255.0)
            if img.max() > 1.0:
                img = img / 255.0
            img = color.rgb2gray(img)
        else:
            img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0

        h, w = img.shape
        r = min(h // 2, w // 2)
        img = img[0:2 * r, 0:2 * r]
        L = img.shape[0]

        # Nails on a circle
        angles = np.linspace(0, 2 * np.pi, NUM_NAILS, endpoint=False)
        rows = (L / 2 + (L * 0.5) * np.cos(angles)).astype(int)
        cols = (L / 2 + (L * 0.5) * np.sin(angles)).astype(int)
        nails = list(zip(rows, cols))  # (row, col)

        canvas = np.ones_like(img)  # white canvas

        # For instructions
        lines = []  # (from, to) pairs, 1-indexed

        # For timelapse
        frames_dir = os.path.join(jd, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        def save_frame(frame_idx: int):
            fname = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close()

        current = 0
        save_frame(0)  # initial blank

        for i in range(NUM_LINES):
            best_j = current
            best_avg = np.inf
            best_rr = best_cc = None

            cr, cc = nails[current]
            for j in range(NUM_NAILS):
                rr, cc2 = draw.line(cr, cc, nails[j][0], nails[j][1])
                rr = np.clip(rr, 0, L - 1)
                cc2 = np.clip(cc2, 0, L - 1)
                avg = float(np.mean(img[rr, cc2]))
                if avg < best_avg:
                    best_avg = avg
                    best_j = j
                    best_rr, best_cc = rr, cc2

            if best_rr is not None:
                img[best_rr, best_cc] = LIGHTEN_TO
                canvas[best_rr, best_cc] = 0.0  # draw black line

            # record line (1-indexed)
            lines.append((current + 1, best_j + 1))
            current = best_j

            if SNAPSHOT_EVERY and ((i + 1) % SNAPSHOT_EVERY == 0):
                save_frame(i + 1)

        # ---------------------
        # Save final PNG
        # ---------------------
        result_png = os.path.join(jd, "string_art_result.png")
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(result_png, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        # ---------------------
        # Save CSV instructions
        # ---------------------
        csv_path = os.path.join(jd, "string_art_lines.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("from_nail,to_nail\n")
            for a, b in lines:
                f.write(f"{a},{b}\n")

        # ---------------------
        # Make PDF from CSV
        # ---------------------
        pdf_path = os.path.join(jd, "string_art_instructions.pdf")
        build_pdf_from_csv(csv_path, pdf_path)

        # ---------------------
        # Make MP4 timelapse
        # ---------------------
        mp4_path = os.path.join(jd, "string_art_timelapse.mp4")
        make_mp4_from_frames(frames_dir, mp4_path)

        # Done
        status.status = "done"
        status.resultImageUrl = build_file_url(job_id, "string_art_result.png")
        status.resultPdfUrl = build_file_url(job_id, "string_art_instructions.pdf")
        status.resultCsvUrl = build_file_url(job_id, "string_art_lines.csv")
        status.resultTimelapseUrl = build_file_url(job_id, "string_art_timelapse.mp4")
        write_status(status)

        # 👇 DEBUG: log success
        print(f"[JOB {job_id}] Finished OK", flush=True)

    except Exception as e:
        status.status = "error"
        status.error = str(e)
        write_status(status)

        # 👇 DEBUG: log the error and stack trace to Render logs
        print(f"[JOB {job_id}] ERROR: {e!r}", flush=True)
        traceback.print_exc()


def build_pdf_from_csv(csv_path: str, pdf_path: str) -> None:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for i, line in enumerate(f, start=1):
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            from_nail, to_nail = parts
            rows.append([i, from_nail, to_nail])

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    story = []
    story.append(Paragraph("<b>String Art Threading Instructions</b>", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Nail 1 is at the top (12 o'clock). Numbers increase clockwise.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    header = ["Step", "From Nail", "To Nail"]
    data = [header] + rows
    col_widths = [25 * mm, 45 * mm, 45 * mm]

    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    story.append(table)
    doc.build(story)


def make_mp4_from_frames(frames_dir: str, mp4_path: str, fps: int = 30) -> None:
    frames = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(".png")
        ]
    )
    if not frames:
        return

    writer = imageio.get_writer(mp4_path, fps=fps)
    try:
        for f in frames:
            im = imageio.imread(f)
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=-1)
            if im.dtype != np.uint8:
                im = np.clip(im * 255, 0, 255).astype(np.uint8)
            writer.append_data(im)
    finally:
        writer.close()


# -------------------------------------------------------------------
# API endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "publicBaseUrl": PUBLIC_BASE_URL or "(relative)",
        "filesRoot": JOBS_ROOT,
    }


@app.post("/redeem-upload", response_model=JobStatus)
async def redeem_upload(file: UploadFile = File(...)):
    """
    Start a job from an uploaded image.
    Always writes a status.json file so /status/{job_id} never 404s.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # 1) Create job folder and input path
    job_id = uuid.uuid4().hex[:12]
    jd = job_dir(job_id)
    os.makedirs(jd, exist_ok=True)

    input_path = os.path.join(jd, "input.jpg")

    # 2) Save the uploaded file
    try:
        contents = await file.read()
        with open(input_path, "wb") as out:
            out.write(contents)
    except Exception as e:
        # If saving fails, still create a status.json with error
        error_status = JobStatus(
            jobId=job_id,
            status="error",
            error=f"Failed to save upload: {e}",
        )
        write_status(error_status)
        return error_status

    # 3) Write initial queued status synchronously
    status = JobStatus(jobId=job_id, status="queued")
    write_status(status)

    # 4) Kick off background processing
    EXECUTOR.submit(generate_string_art_assets, input_path, job_id)

    # 5) Return the queued status to the caller
    return status

@app.post("/redeem-upload", response_model=JobStatus)
async def redeem_upload(file: UploadFile = File(...)):
    """
    Start a job from an uploaded image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    job_id = uuid.uuid4().hex[:12]
    jd = job_dir(job_id)
    os.makedirs(jd, exist_ok=True)

    input_path = os.path.join(jd, "input.jpg")
    try:
        contents = await file.read()
        with open(input_path, "wb") as out:
            out.write(contents)
    except Exception as e:
        status = JobStatus(jobId=job_id, status="error", error=f"Failed to save upload: {e}")
        write_status(status)
        return status

    status = JobStatus(jobId=job_id, status="queued")
    write_status(status)

    EXECUTOR.submit(generate_string_art_assets, input_path, job_id)

    return status


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    status = read_status(job_id)

    # If already done, ensure URLs are absolute
    if status.status == "done":
        if status.resultImageUrl is None:
            status.resultImageUrl = build_file_url(job_id, "string_art_result.png")
        if status.resultPdfUrl is None:
            status.resultPdfUrl = build_file_url(job_id, "string_art_instructions.pdf")
        if status.resultCsvUrl is None:
            status.resultCsvUrl = build_file_url(job_id, "string_art_lines.csv")
        if status.resultTimelapseUrl is None:
            status.resultTimelapseUrl = build_file_url(job_id, "string_art_timelapse.mp4")

    return status


@app.get("/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    path = os.path.join(job_dir(job_id), filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)