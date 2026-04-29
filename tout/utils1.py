# utils.py
import cv2, numpy as np
from scipy.signal import find_peaks, savgol_filter
import math

# --- image helpers ---
def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- grid detection for calibration pixels_per_mm ---
def estimate_pixels_per_mm(img):
    # detect grid lines by red channel prominence
    red = img[:,:,2].astype(float)
    enh = red - 0.5*(img[:,:,0].astype(float)+img[:,:,1].astype(float))
    enh = np.clip(enh, 0, None)
    vert = np.mean(enh, axis=1)
    horiz = np.mean(enh, axis=0)
    # find peaks
    peaks_v = np.where((vert[1:-1] > vert[:-2]) & (vert[1:-1] > vert[2:]))[0]+1
    diffs = np.diff(peaks_v) if len(peaks_v)>2 else np.array([40])
    spacing = int(np.median(diffs)) if len(diffs)>0 else 40
    # guessing spacing corresponds to 5mm (large lines) => pixels per mm
    pixels_per_mm = spacing/5.0
    return pixels_per_mm

# --- segmentation heuristic (fallback if no U-Net) ---
def segment_trace_heuristic(lead_img):
    gray = to_gray(lead_img)
    # equalize and adaptive threshold (black trace => white mask)
    eq = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,25,10)
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return clean

# --- convert mask to 1D signal: median y per_column ---
def mask_to_signal(mask, smooth=True):
    H,W = mask.shape
    yvals = np.zeros(W, dtype=float)
    for x in range(W):
        ys = np.where(mask[:,x] > 0)[0]
        if ys.size>0:
            yvals[x] = np.median(ys)
        else:
            yvals[x] = np.nan
    # interpolate nans
    nans = np.isnan(yvals)
    if np.any(~nans):
        yvals[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], yvals[~nans])
    yvals = (H - yvals)  # invert: upward positive
    if smooth and W>51:
        k = min(101, (W//20)*2+1)
        yvals = savgol_filter(yvals, k, 3)
    return yvals

# --- R peak detection (Pan-Tompkins-like simplified) ---
def detect_r_peaks(signal, pixels_per_mm, fs=None):
    # infer sampling frequency from pixels_per_mm using 25 mm/s
    if fs is None:
        dt = 0.04 / max(1.0, pixels_per_mm)
        fs = 1.0/dt
    # normalize
    sig = (signal - np.mean(signal)) / (np.std(signal)+1e-9)
    # find_peaks with prominence
    min_dist = int(0.25 * fs)  # 250ms
    peaks, props = find_peaks(sig, height=0.4, distance=min_dist, prominence=0.5)
    if peaks.size==0:
        peaks, props = find_peaks(sig, height=0.2, distance=min_dist, prominence=0.3)
    return peaks, props, fs

# --- delineation heuristics: find QRS onset/offset, P onset, T offset ---
def delineate_beats(signal, peaks, fs):
    onsets = []
    offsets = []
    P_onsets = []
    T_offsets = []
    N = len(signal)
    for p in peaks:
        # window around p
        w = int(0.08*fs)  # 80ms
        left = max(0, p - w)
        right = min(N-1, p + w)
        window = signal[left:right+1]
        peak_val = signal[p]
        thr = 0.25 * peak_val
        # onset: last index before peak where abs(val) < thr
        rel = np.where(np.abs(window[:p-left]) < thr)[0]
        if rel.size>0:
            onset = left + rel[-1]
        else:
            onset = left
        # offset: first index after peak where abs(val) < thr
        rel2 = np.where(np.abs(window[p-left:]) < thr)[0]
        if rel2.size>0:
            offset = p + rel2[0]
        else:
            offset = right
        onsets.append(onset)
        offsets.append(offset)
        # P onset: look 200ms before onset
        p_search = int(0.2 * fs)
        s0 = max(0, onset - p_search)
        pre = signal[s0:onset]
        # rough: P where derivative increases
        dpre = np.diff(pre)
        p_idx = s0 + np.argmax(dpre) if dpre.size>0 else s0
        P_onsets.append(p_idx)
        # T offset: look 400ms after offset
        t_search = int(0.4 * fs)
        s1 = offset
        post = signal[s1: min(N, s1 + t_search)]
        # T offset approximate where abs < 0.1*peak_std
        th = 0.15 * np.max(np.abs(signal))
        rel3 = np.where(np.abs(post) < th)[0]
        if rel3.size>0:
            t_off = s1 + rel3[0]
        else:
            t_off = min(N-1, s1 + int(0.2*fs))
        T_offsets.append(t_off)
    return np.array(P_onsets), np.array(onsets), np.array(offsets), np.array(T_offsets)

# --- compute intervals in ms ---
def compute_intervals_from_delineation(P_on, Q_on, Q_off, T_off, fs):
    dt = 1.0/fs
    PRs = (Q_on - P_on) * dt * 1000.0
    QRSs = (Q_off - Q_on) * dt * 1000.0
    QTs = (T_off - Q_on) * dt * 1000.0
    # RR median in seconds
    if len(Q_on) >= 2:
        RR = np.median(np.diff(Q_on)) * dt
    else:
        RR = np.nan
    # Bazett QTc
    QTc = QTs / math.sqrt(RR) if (not math.isnan(RR) and RR>0) else np.nan
    return {
        "PR_ms":  float(np.nanmedian(PRs))  if len(PRs)  > 0 else np.nan,
        "QRS_ms": float(np.nanmedian(QRSs)) if len(QRSs) > 0 else np.nan,
        "QT_ms":  float(np.nanmedian(QTs))  if len(QTs)  > 0 else np.nan,
}

# Note: compute_intervals_from_delineation above is intentionally simple; we compute medians below properly.
