# tl_end2end.py
# End-to-end: split dataset -> train transfer learning -> save artifacts -> webcam demo + robustness test

import os, sys, shutil, random, argparse, json, pathlib
from collections import defaultdict

import numpy as np
import cv2
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def list_images(folder):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    out = []
    for root,_,files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                out.append(os.path.join(root,f))
    return out

# -----------------------------
# 1) Split dataset raw -> data/train|val|test
# -----------------------------
def split_dataset(raw_root, data_root, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    classes = sorted([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root,d))])
    if not classes:
        print("[error] tidak menemukan subfolder kelas di", raw_root); sys.exit(1)

    for split in ["train","val","test"]:
        for c in classes:
            ensure_dir(os.path.join(data_root, split, c))

    print("[info] mulai split per kelas 70 15 15")
    for c in classes:
        src = os.path.join(raw_root, c)
        imgs = list_images(src)
        if len(imgs) == 0:
            print(f"[warning] kelas {c} kosong, lewati"); continue
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        n_test  = n - n_train - n_val

        if n_train == 0 and n >= 1: n_train = 1
        if n_val   == 0 and n - n_train >= 2: n_val = 1
        if n_test  == 0 and n - n_train - n_val >= 1: n_test = 1

        parts = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train+n_val],
            "test":  imgs[n_train+n_val:],
        }
        for split, paths in parts.items():
            dst_dir = os.path.join(data_root, split, c)
            for p in paths:
                shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
        print(f"[ok] {c}  total {n}  -> train {len(parts['train'])}  val {len(parts['val'])}  test {len(parts['test'])}")

    print("[done] split selesai di", data_root)

# -----------------------------
# 2) Data loaders (pin_memory kondisional)
# -----------------------------
def build_loaders(data_root, batch=32):
    use_gpu = torch.cuda.is_available()
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.25,0.25,0.25,0.05),
        transforms.GaussianBlur(3, sigma=(0.1,1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    ds = {}
    for split, tf in [("train",train_tf),("val",eval_tf),("test",eval_tf)]:
        p = os.path.join(data_root, split)
        if not os.path.isdir(p):
            print(f"[error] folder {p} tidak ada"); sys.exit(1)
        ds[split] = datasets.ImageFolder(p, transform=tf)

    classes = ds["train"].classes
    sizes = {k: len(v) for k,v in ds.items()}

    # Windows + CPU cenderung lebih stabil workers=0
    workers = 0 if (os.name == "nt" and not use_gpu) else 2
    pin_mem = use_gpu

    dl = {
        k: DataLoader(v, batch_size=batch, shuffle=(k=="train"),
                      num_workers=workers, pin_memory=pin_mem)
        for k,v in ds.items()
    }

    print("[info] kelas:", classes)
    print("[info] jumlah train", sizes["train"], "val", sizes["val"], "test", sizes["test"])
    return dl, sizes, classes

# -----------------------------
# 3) Model
# -----------------------------
def build_model(name, nclass):
    if name=="mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        in_ch = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_ch, nclass)
        return m
    elif name=="resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_ch = m.fc.in_features
        m.fc = nn.Linear(in_ch, nclass)
        return m
    elif name=="efficientnetb0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_ch = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_ch, nclass)
        return m
    else:
        raise ValueError("nama model tidak dikenali")

def freeze_backbone(mdl, freeze=True):
    for n,p in mdl.named_parameters():
        if ("classifier" in n) or (".fc" in n):
            p.requires_grad = True
        else:
            p.requires_grad = not freeze

def partial_unfreeze_top(mdl, name="mobilenetv2"):
    for n,p in mdl.named_parameters():
        p.requires_grad = True
        if name=="mobilenetv2":
            if any(f"features.{i}." in n for i in range(0,7)):
                p.requires_grad = False
        elif name=="resnet50":
            if "layer1" in n:
                p.requires_grad = False
        elif name=="efficientnetb0":
            if any(f"features.{i}." in n for i in range(0,3)):
                p.requires_grad = False

# -----------------------------
# 4) Train + Eval
# -----------------------------
def train_and_eval(data_root, model_name="mobilenetv2", batch=32, ep1=8, ep2=8, lr1=1e-3, lr2=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders, sizes, classes = build_loaders(data_root, batch=batch)
    model = build_model(model_name, len(classes)).to(device)
    crit = nn.CrossEntropyLoss()

    def epoch(phase, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        loss_sum = 0.0; correct = 0
        for x,y in loaders[phase]:
            x,y = x.to(device), y.to(device)
            if is_train: optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(is_train):
                logits = model(x)
                loss = crit(logits, y)
                if is_train:
                    loss.backward(); optimizer.step()
            loss_sum += loss.item()*x.size(0)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
        return loss_sum/sizes[phase], correct/sizes[phase]

    # Stage 1 freeze backbone
    freeze_backbone(model, True)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr1)
    best_w = None; best_acc = 0.0
    print("[stage1] freeze backbone latih classifier")
    for e in range(1, ep1+1):
        tl,ta = epoch("train", opt)
        vl,va = epoch("val")
        if va>best_acc: best_acc=va; best_w = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"  ep {e}/{ep1}  train_acc {ta:.3f}  val_acc {va:.3f}")
    if best_w: model.load_state_dict(best_w)

    # Stage 2 fine-tune top
    partial_unfreeze_top(model, model_name)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2, weight_decay=1e-4)
    print("[stage2] fine-tune sebagian layer atas")
    for e in range(1, ep2+1):
        tl,ta = epoch("train", opt)
        vl,va = epoch("val")
        if va>best_acc: best_acc=va; best_w = {k:v.cpu() for k,v in model.state_dict().items()}
        print(f"  ep {e}/{ep2}  train_acc {ta:.3f}  val_acc {va:.3f}")
    if best_w: model.load_state_dict(best_w)

    # Test metrics
    model.eval()
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for x,y in loaders["test"]:
            logits = model(x.to(device))
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save artifacts
    ensure_dir("artifacts")
    torch.save(model.state_dict(), os.path.join("artifacts","model_best.pt"))
    with open(os.path.join("artifacts","classes.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(classes))
    with open(os.path.join("artifacts","report.json"),"w",encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join("artifacts","confusion_matrix.png"))
    plt.close(fig)

    print("\n=== TEST METRICS (macro) ===")
    print(f"accuracy  {report['accuracy']:.4f}")
    m = report["macro avg"]
    print(f"precision {m['precision']:.4f}  recall {m['recall']:.4f}  f1 {m['f1-score']:.4f}")
    print("[save] artifacts/model_best.pt  artifacts/classes.txt  artifacts/confusion_matrix.png  artifacts/report.json")

    # Robustness test
    robustness_eval(data_root, model, device)

# -----------------------------
# 4b) Robustness evaluation
# -----------------------------
def _add_gauss_noise_tensor(t, std=0.07):
    n = torch.randn_like(t) * std
    return torch.clamp(t + n, 0.0, 1.0)

def _occlude_tensor(t, size=56):
    _, h, w = t.shape
    y = random.randint(0, max(0, h - size))
    x = random.randint(0, max(0, w - size))
    t[:, y:y+size, x:x+size] = 0.0
    return t

def _make_transform(kind):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    base = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    if kind=="blur":
        base.insert(2, transforms.GaussianBlur(5))
    if kind=="noise":
        base.append(transforms.Lambda(lambda t: _add_gauss_noise_tensor(t, std=0.07)))
    if kind=="occl":
        base.append(transforms.Lambda(lambda t: _occlude_tensor(t, size=56)))
    base.append(transforms.Normalize(mean,std))
    return transforms.Compose(base)

def robustness_eval(data_root, model, device):
    test_root = os.path.join(data_root,"test")
    raw_ds = datasets.ImageFolder(test_root, transform=lambda img: img)

    def run_with_transform(tf):
        y_true=[]; y_pred=[]
        model.eval()
        with torch.no_grad():
            for img, y in raw_ds:
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                y_true.append(y)
                y_pred.append(int(logits.argmax(1).item()))
        rep = classification_report(y_true,y_pred,output_dict=True,zero_division=0)
        return float(rep["accuracy"]), float(rep["macro avg"]["f1-score"])

    results = {}
    for kind in ["clean","blur","noise","occl"]:
        tf = _make_transform(kind)
        acc, f1 = run_with_transform(tf)
        results[kind] = {"accuracy": acc, "macro_f1": f1}
        print(f"[robust] {kind}  acc {acc:.4f}  f1 {f1:.4f}")

    ensure_dir("artifacts")
    with open("artifacts/robustness.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # bar chart
    labels = list(results.keys())
    accs = [results[k]["accuracy"] for k in labels]
    f1s  = [results[k]["macro_f1"] for k in labels]
    xpos = np.arange(len(labels)); width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(xpos - width/2, accs, width, label="Accuracy")
    plt.bar(xpos + width/2, f1s,  width, label="Macro F1")
    plt.xticks(xpos, labels); plt.ylim(0,1.0); plt.title("Robustness Test"); plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/robustness.png")
    plt.close()
    print("[save] artifacts/robustness.json  artifacts/robustness.png")

# -----------------------------
# 5) Webcam demo
# -----------------------------
def webcam_demo(model_name="mobilenetv2", classes_path="artifacts/classes.txt", weights_path="artifacts/model_best.pt", cam_index=0):
    if not (os.path.exists(classes_path) and os.path.exists(weights_path)):
        print("[error] artifacts tidak lengkap. latih model terlebih dahulu atau pastikan path benar")
        return
    classes = [l.strip() for l in open(classes_path,"r",encoding="utf-8").read().splitlines() if l.strip()]

    # build model tanpa pretrained lalu load weights
    if model_name=="mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    elif model_name=="resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(classes))
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(classes))
    state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ema = None; alpha = 0.3

    print("Webcam demo. Tekan Q untuk keluar")
    while True:
        ok, frame = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tf(rgb).unsqueeze(0)
        with torch.no_grad():
            logits = m(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
        if ema is None: ema = prob
        else: ema = alpha*prob + (1-alpha)*ema
        top = int(np.argmax(ema)); conf = float(ema[top])
        text = f"{classes[top]}  {conf*100:.1f}%"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80,255,80), 2)
        cv2.imshow("Webcam Classification", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

# -----------------------------
# 6) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=None, help="folder foto mentah per kelas")
    ap.add_argument("--data", type=str, default="data", help="root dataset dengan train val test")
    ap.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2","resnet50","efficientnetb0"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ep1", type=int, default=8)
    ap.add_argument("--ep2", type=int, default=8)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--webcam", action="store_true", help="jalankan demo webcam setelah training")
    ap.add_argument("--webcam-only", action="store_true", help="lewati training langsung demo webcam")
    ap.add_argument("--cam-index", type=int, default=0)
    args = ap.parse_args()

    if args.webcam_only:
        webcam_demo(args.model, cam_index=args.cam_index)
        return

    if args.raw:
        split_dataset(args.raw, args.data, 0.70, 0.15, 0.15)

    train_and_eval(args.data, args.model, args.batch, args.ep1, args.ep2, args.lr1, args.lr2)
    if args.webcam:
        webcam_demo(args.model, cam_index=args.cam_index)

if __name__ == "__main__":
    main()


#py mainprogram.py --data data --model efficientnetb0 --ep1 2 --ep2 2 --batch 8
#py mainprogram.py --webcam-only --model efficientnetb0