# pipeline/string_art_timelapse.py
import os, numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, draw
from pathlib import Path
try:
    import imageio.v2 as imageio
    import imageio_ffmpeg  # noqa
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

def make_timelapse(image_path: str, out_dir: str,
                   num_nails=240, num_lines=1500, lighten_to=1.0,
                   snapshot_every=25, mp4=True, gif=False, fps_mp4=30, fps_gif=20):
    """
    Re-runs greedy draw to render frames and video. Returns:
      - timelapse_mp4 (optional)
      - timelapse_gif (optional)
      - final_png
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR = out / "frames"
    FRAME_PREFIX = "frame_"
    FINAL_PNG = out / "string_art_result.png"
    MP4_FILE = out / "string_art_timelapse.mp4"
    GIF_FILE = out / "string_art_timelapse.gif"

    def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

    def save_frame(canvas, idx):
        fname = FRAMES_DIR / f"{FRAME_PREFIX}{idx:05d}.png"
        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        plt.axis("off"); plt.tight_layout(pad=0)
        plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    img = io.imread(image_path)
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

    h, w = img.shape
    r = min(h // 2, w // 2)
    img = img[0:2 * r, 0:2 * r]
    L = img.shape[0]

    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)
    rows = (L / 2 + (L * 0.5) * np.cos(angles)).astype(int)
    cols = (L / 2 + (L * 0.5) * np.sin(angles)).astype(int)
    nails = list(zip(rows, cols))

    canvas = np.ones_like(img)
    if snapshot_every and snapshot_every > 0:
        ensure_dir(FRAMES_DIR); save_frame(canvas, 0)

    current = 0
    for i in range(num_lines):
        best_j = current; best_avg = np.inf
        best_rr = best_cc = None
        cr, cc = nails[current]
        for j in range(num_nails):
            rr, cc2 = draw.line(cr, cc, nails[j][0], nails[j][1])
            rr = np.clip(rr, 0, L - 1); cc2 = np.clip(cc2, 0, L - 1)
            avg = float(np.mean(img[rr, cc2]))
            if avg < best_avg:
                best_avg = avg; best_j = j; best_rr, best_cc = rr, cc2
        if best_rr is not None:
            img[best_rr, best_cc] = lighten_to
            canvas[best_rr, best_cc] = 0.0

        current = best_j
        if snapshot_every and ((i + 1) % snapshot_every == 0):
            save_frame(canvas, i + 1)

    # final png
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(FINAL_PNG, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    out_paths = {"final_png": str(FINAL_PNG)}

    if HAVE_IMAGEIO:
        frames = sorted([str((FRAMES_DIR / f)) for f in os.listdir(FRAMES_DIR) if f.startswith(FRAME_PREFIX)])
        if gif:
            imgs = [imageio.imread(f) for f in frames]
            imageio.mimsave(GIF_FILE, imgs, fps=fps_gif)
            out_paths["timelapse_gif"] = str(GIF_FILE)
        if mp4:
            writer = imageio.get_writer(str(MP4_FILE), format="FFMPEG", fps=fps_mp4, codec="libx264", quality=8)
            try:
                for f in frames:
                    im = imageio.imread(f)
                    if im.ndim == 2:
                        im = np.stack([im, im, im], axis=-1)
                    if im.dtype != np.uint8:
                        im = np.clip(im, 0, 255).astype(np.uint8)
                    writer.append_data(im)
            finally:
                writer.close()
            out_paths["timelapse_mp4"] = str(MP4_FILE)

    return out_paths