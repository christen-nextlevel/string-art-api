# pipeline/string_art_matplotlib.py
import numpy as np
from skimage import io, color, draw
import matplotlib.pyplot as plt
from pathlib import Path

def generate_string_art(image_path: str, out_dir: str,
                        num_nails=240, num_lines=1300, lighten_to=1.0):
    """
    Returns dict with:
      - result_png
      - lines_csv
      - instr_txt
      - nails_count
      - lines_count
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = out / "string_art_result.png"
    INSTR_TXT   = out / "string_art_instructions.txt"
    LINES_CSV   = out / "string_art_lines.csv"

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
            avg = float(np.mean(img[rr, cc2]))
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

    # save result
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
    plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    with open(INSTR_TXT, "w", encoding="utf-8") as f:
        f.write("STRING ART THREAD ORDER (1-indexed, nail 1 at 12 o'clock):\n")
        for i in range(0, len(order), 25):
            f.write(" ".join(str(x) for x in order[i:i+25]) + "\n")

    with open(LINES_CSV, "w", encoding="utf-8") as f:
        f.write("from_nail,to_nail\n")
        for a, b in lines:
            f.write(f"{a},{b}\n")

    return {
        "result_png": str(OUTPUT_FILE),
        "lines_csv": str(LINES_CSV),
        "instr_txt": str(INSTR_TXT),
        "nails_count": num_nails,
        "lines_count": num_lines
    }