import os, glob, cv2, argparse, pandas as pd
from hdr.deghosting import merge_mertens_deghosted
from hdr.metrics import calculate_shannon_entropy, calculate_clipping_percentages

def process_directory(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subdirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        subdirs = [in_dir]

    report = []
    for s in subdirs:
        name = os.path.basename(s)
        files = sorted(glob.glob(os.path.join(s, "*.jpg")) + glob.glob(os.path.join(s, "*.png")))
        if len(files) < 2:
            continue
        imgs = [cv2.imread(f) for f in files]
        hdr, mask = merge_mertens_deghosted(imgs)
        out_p = os.path.join(out_dir, f"{name}_HDR.jpg")
        cv2.imwrite(out_p, hdr)
        ent = calculate_shannon_entropy(hdr)
        u, o = calculate_clipping_percentages(hdr)
        report.append({"Сцена": name, "Кадров": len(files), "Энтропия": round(ent, 3), "Клиппинг": round(u+o, 2)})

    df = pd.DataFrame(report)
    df.to_csv(os.path.join(out_dir, "batch_report.csv"), index=False)
    print("Пакетная обработка завершена.")
    print(df.to_string())

if __name__ == "__main__":
    process_directory("data", "batch_output")
