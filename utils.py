import cv2, numpy as np
from scipy.signal import savgol_filter, resample

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def estimate_pixels_per_mm(img):
    red = img[:,:,2].astype(float)
    enh = red - 0.5*(img[:,:,0].astype(float)+img[:,:,1].astype(float))
    enh = np.clip(enh, 0, None)
    vert = np.mean(enh, axis=1)
    peaks = np.where((vert[1:-1] > vert[:-2]) & (vert[1:-1] > vert[2:]))[0]+1
    diffs = np.diff(peaks) if len(peaks) > 2 else np.array([40])
    spacing = int(np.median(diffs))
    return spacing / 5.0  # pixels / mm

def split_12leads(img):
    h, w = img.shape[:2]
    rows, cols = 3, 4
    leads = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*h//rows, (r+1)*h//rows
            x0, x1 = c*w//cols, (c+1)*w//cols
            pad = int(0.02 * min(h//rows, w//cols))
            leads.append(img[y0+pad:y1-pad, x0+pad:x1-pad])
    return leads

def mask_to_signal(mask):
    H, W = mask.shape
    y = np.zeros(W)
    for x in range(W):
        ys = np.where(mask[:,x] > 0)[0]
        y[x] = np.median(ys) if len(ys) else np.nan
    nans = np.isnan(y)
    y[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], y[~nans])
    y = H - y
    if W > 51:
        y = savgol_filter(y, 51, 3)
    return y

def pixels_to_mV(signal_px, pixels_per_mm):
    baseline = np.median(signal_px)
    mm = (signal_px - baseline) / pixels_per_mm
    return mm * 0.1  # 10 mm / mV

def resample_signal(signal, fs, duration):
    n = int(np.floor(fs * duration))
    return resample(signal, n)

def segment_with_unet(model, lead_img, device):
    gray = cv2.cvtColor(lead_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (512,512))
    x = torch.tensor(img/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        y = model(x)[0,0].cpu().numpy()
    y = cv2.resize(y, (lead_img.shape[1], lead_img.shape[0]))
    return (y > 0.5).astype(np.uint8) * 255
