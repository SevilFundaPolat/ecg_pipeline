def segment_with_unet(model, lead_img, device):
    gray = cv2.cvtColor(lead_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (512,512))
    x = torch.tensor(img/255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        y = model(x)[0,0].cpu().numpy()
    y = cv2.resize(y, (lead_img.shape[1], lead_img.shape[0]))
    return (y > 0.5).astype(np.uint8) * 255

import torch, pandas as pd, os, cv2
from utils_ecg import *

def infer_single_image(img_path, fs, model, device):
    img = cv2.imread(img_path)
    ppm = estimate_pixels_per_mm(img)
    leads_imgs = split_12leads(img)

    signals = {}
    for lead, lead_img in zip(LEAD_NAMES, leads_imgs):
        mask = segment_with_unet(model, lead_img, device)
        sig_px = mask_to_signal(mask)
        sig_mv = pixels_to_mV(sig_px, ppm)
        duration = 10.0 if lead == "II" else 2.5
        sig_mv = resample_signal(sig_mv, fs, duration)
        signals[lead] = sig_mv
    return signals

def build_submission(test_csv, image_dir, model, device):
    test = pd.read_csv(test_csv)
    rows = []

    for _, r in test.iterrows():
        base_id = r["id"]
        fs = r["fs"]
        img_path = os.path.join(image_dir, f"{base_id}.png")

        signals = infer_single_image(img_path, fs, model, device)

        for lead, sig in signals.items():
            for i, v in enumerate(sig):
                rows.append({
                    "id": f"{base_id}_{i}_{lead}",
                    "base_id": base_id,
                    "lead": lead,
                    "row_id": i,
                    "value": float(v)
                })

    df = pd.DataFrame(rows)
    df.to_parquet("submission.parquet", index=False)
    print("submission.parquet créé ✔")
