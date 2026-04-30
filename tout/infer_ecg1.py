# infer_ecg.py
import os, cv2, numpy as np, pandas as pd, math
from utils import estimate_pixels_per_mm, segment_trace_heuristic, mask_to_signal, detect_r_peaks, delineate_beats
from models.unet import UNet
import torch
import matplotlib.pyplot as plt

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def load_unet_checkpoint(path, device):
    model = UNet(in_ch=1, out_ch=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def segment_with_unet(model, lead_img, device):
    gray = cv2.cvtColor(lead_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (512,512))
    inp = torch.tensor(img/255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)[0,0].cpu().numpy()
    pred = (pred*255).astype('uint8')
    pred = cv2.resize(pred, (lead_img.shape[1], lead_img.shape[0]))
    _, binary = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
    return binary

def split_12leads(img):
    h,w = img.shape[:2]
    rows, cols = 3, 4
    h_step = h // rows
    w_step = w // cols
    leads = []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, r*h_step)
            y1 = min(h, (r+1)*h_step)
            x0 = max(0, c*w_step)
            x1 = min(w, (c+1)*w_step)
            pad = int(0.02*min(h_step,w_step))
            leads.append(img[y0+pad:y1-pad, x0+pad:x1-pad])
    return leads

def compute_intervals_for_signal(signal, pixels_per_mm):
    # detect R peaks
    peaks, props, fs = detect_r_peaks(signal, pixels_per_mm)
    if peaks.size==0:
        return dict(PR_ms=np.nan,QRS_ms=np.nan,QT_ms=np.nan,QTc_ms=np.nan, n_beats=0)
    P_on, Q_on, Q_off, T_off = delineate_beats(signal, peaks, fs)
    # compute medians
    dt = 1.0/fs
    PRs = (Q_on - P_on) * dt * 1000.0
    QRSs = (Q_off - Q_on) * dt * 1000.0
    QTs = (T_off - Q_on) * dt * 1000.0
    median_PR = float(np.nanmedian(PRs)) if PRs.size>0 else np.nan
    median_QRS = float(np.nanmedian(QRSs)) if QRSs.size>0 else np.nan
    median_QT = float(np.nanmedian(QTs)) if QTs.size>0 else np.nan
    # RR for QTc (Bazett)
    if peaks.size>1:
        RR_sec = np.median(np.diff(peaks)) * dt
        QTc = median_QT / math.sqrt(RR_sec) if RR_sec>0 else np.nan
    else:
        QTc = np.nan
    return dict(PR_ms=round(median_PR) if not np.isnan(median_PR) else np.nan,
                QRS_ms=round(median_QRS) if not np.isnan(median_QRS) else np.nan,
                QT_ms=round(median_QT) if not np.isnan(median_QT) else np.nan,
                QTc_ms=round(QTc) if not np.isnan(QTc) else np.nan,
                n_beats=int(peaks.size))

def infer_image(path_image, unet_ckpt=None, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(path_image)
    pixels_per_mm = estimate_pixels_per_mm(img)
    print("Pixels per mm:", pixels_per_mm)
    leads_imgs = split_12leads(img)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if unet_ckpt and os.path.exists(unet_ckpt):
        print("Loading U-Net checkpoint:", unet_ckpt)
        model = load_unet_checkpoint(unet_ckpt, device)
    results = []
    for i, lead_img in enumerate(leads_imgs):
        name = LEAD_NAMES[i]
        if model:
            mask = segment_with_unet(model, lead_img, device)
        else:
            mask = segment_trace_heuristic(lead_img)
        signal = mask_to_signal(mask)
        intervals = compute_intervals_for_signal(signal, pixels_per_mm)
        intervals['lead'] = name
        results.append(intervals)
        # save plot for quick view
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,2))
        ax.plot(signal)
        ax.set_title(f"{name} | PR:{intervals['PR_ms']} QRS:{intervals['QRS_ms']} QT:{intervals['QT_ms']} QTc:{intervals['QTc_ms']}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"signal_{name}.png"))
        plt.close(fig)
    df = pd.DataFrame(results)[['lead','PR_ms','QRS_ms','QT_ms','QTc_ms','n_beats']]
    csv_out = os.path.join(out_dir, os.path.basename(path_image).rsplit('.',1)[0] + "_intervals.csv")
    df.to_csv(csv_out, index=False)
    print("Saved:", csv_out)
    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True)
    parser.add_argument('--ckpt', default=None, help='U-Net checkpoint path (optional)')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    df = infer_image(args.img, args.ckpt, args.out)
    print(df)
