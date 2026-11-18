"""
ParkSense - generative AI component (Week 8): unpaired image-to-image translation
with a CycleGAN (Zhu et al., 2017), trained from scratch on CPU / free Colab.

  Domain A = UFPR04 dry-weather frames (Sunny + Cloudy, training days only)
  Domain B = UFPR04 RAINY frames from the 'gan_pool' days - used WITHOUT labels.
             (Rain footage from your own camera is free; labelling it is not.)

The translator keeps the scene geometry, so every slot polygon and its label still
apply to the translated frame -> labelled synthetic rain for free.

Resolution trick: the GAN works at quarter resolution (320x180, 96x96 patches) to fit
a CPU budget; at inference the high-frequency detail of the original frame is added
back (Laplacian detail transfer), so slot crops stay sharp.

  python training/cyclegan.py train --minutes 45
  python training/cyclegan.py translate --split train   # -> data/gen_cyclegan_rain.npz (labelled crops)
  python training/cyclegan.py translate --split test    # -> data/stress_cyclegan_rain.npz (stress test)
"""
import argparse, csv, json, os, sys, time, collections, itertools
import numpy as np, cv2, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from parksense_core import rectify  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "runs", "cyclegan")
SCALE = 0.25


# ----------------------------------------------------------------------------- networks
class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
                               nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class Gen(nn.Module):
    def __init__(self, ngf=24, n_res=5):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):
            L += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]; c *= 2
        L += [Res(c) for _ in range(n_res)]
        for _ in range(2):
            # resize-convolution instead of ConvTranspose2d: avoids checkerboard artefacts (Odena et al., 2016)
            L += [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1), nn.Conv2d(c, c // 2, 3),
                  nn.InstanceNorm2d(c // 2), nn.ReLU(True)]; c //= 2
        last = nn.Conv2d(c, 3, 7)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)   # starts as the identity mapping
        L += [nn.ReflectionPad2d(3), last, nn.Tanh()]
        self.m = nn.Sequential(*L)

    def forward(self, x):
        # Residual generator (as in SimGAN, Shrivastava et al., 2017): output = input + learned change.
        # Scene content - and therefore the bay labels - is preserved by construction; the GAN only
        # has to learn what rain changes (wet, darker asphalt, reflections, lower contrast).
        return torch.clamp(x + self.m(x), -1, 1)


class Disc(nn.Module):  # PatchGAN
    """No InstanceNorm: on 64x64 crops instance normalisation removes each crop's global
    brightness/saturation - exactly the cues that separate rain from dry - and the
    discriminator went blind (loss stuck at 0.5). Spectral norm keeps it stable instead."""
    def __init__(self, ndf=32):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.m = nn.Sequential(
            sn(nn.Conv2d(3, ndf, 4, 2, 1)), nn.LeakyReLU(.2, True),
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.LeakyReLU(.2, True),
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.LeakyReLU(.2, True),
            sn(nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 1)), nn.LeakyReLU(.2, True),
            sn(nn.Conv2d(ndf * 4, 1, 4, 1, 1)))

    def forward(self, x):
        return self.m(x)


# ----------------------------------------------------------------------------- data
def manifest():
    return list(csv.DictReader(open("data/manifest.csv")))


def load_small(path):
    im = cv2.imread(os.path.join("data/images", path))
    return cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def to_t(im):  # BGR uint8 -> 1x3xHxW in [-1,1] (RGB)
    x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def from_t(t):
    x = ((t.squeeze(0).detach().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def patches(imgs, n, size, rng):
    out = []
    for _ in range(n):
        im = imgs[rng.integers(len(imgs))]
        y = rng.integers(0, im.shape[0] - size); x = rng.integers(0, im.shape[1] - size)
        p = im[y:y + size, x:x + size]
        if rng.random() < .5: p = p[:, ::-1]
        out.append(to_t(np.ascontiguousarray(p)))
    return torch.cat(out)


class Pool:  # image history buffer (Shrivastava et al.) - stabilises the discriminators
    def __init__(self, n=50): self.n, self.d = n, []
    def __call__(self, x):
        if len(self.d) < self.n: self.d.append(x.detach()); return x
        if np.random.rand() < .5:
            i = np.random.randint(self.n); old = self.d[i]; self.d[i] = x.detach(); return old
        return x


def crop_batch(X, n, rng):
    idx = rng.integers(0, len(X), n); out = []
    for i in idx:
        p = X[i]
        if rng.random() < .5: p = p[:, ::-1]
        p = np.rot90(p, rng.integers(4))
        out.append(to_t(np.ascontiguousarray(p)))
    return torch.cat(out)


# ----------------------------------------------------------------------------- train
def train(a):
    torch.manual_seed(0); rng = np.random.default_rng(0); torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    if a.level == "crop":
        d = np.load("data/crops.npz")
        A = d["X"][(d["site"] == "UFPR04") & (d["split"] == "train")]
        B = d["X"][(d["site"] == "UFPR04") & (d["split"] == "gan_pool")]   # labels never read
        sample = lambda X: crop_batch(X, a.bs, rng)
        print(f"crop level: domain A (dry) {len(A)} crops, domain B (rain, unlabelled) {len(B)} crops", flush=True)
    else:
        m = manifest()
        A = [load_small(r["file"]) for r in m if r["site"] == "UFPR04" and r["split"] == "train"]
        B = [load_small(r["file"]) for r in m if r["site"] == "UFPR04" and r["split"] == "gan_pool"]
        sample = lambda X: patches(X, a.bs, a.size, rng)
        print(f"frame level: domain A (dry) {len(A)} frames, domain B (rain, unlabelled) {len(B)} frames", flush=True)
    G, Fb, DA, DB = Gen(), Gen(), Disc(), Disc()
    oG = torch.optim.Adam(itertools.chain(G.parameters(), Fb.parameters()), 2e-4, betas=(.5, .999))
    oD = torch.optim.Adam(itertools.chain(DA.parameters(), DB.parameters()), 2e-4, betas=(.5, .999))
    mse, l1 = nn.MSELoss(), nn.L1Loss(); pA, pB = Pool(), Pool()
    os.makedirs(OUT, exist_ok=True)
    t0, it, log = time.time(), 0, []
    budget = a.minutes * 60
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        lr = 2e-4 * (1.0 if frac < .5 else 2 * (1 - frac))  # linear decay in 2nd half
        for o in (oG, oD):
            for g in o.param_groups: g["lr"] = lr
        a_ = sample(A); b_ = sample(B)
        fb, fa = G(a_), Fb(b_)
        pfb, pfa = DB(fb), DA(fa)
        lg = (mse(pfb, torch.ones_like(pfb)) + mse(pfa, torch.ones_like(pfa))
              + 10 * (l1(Fb(fb), a_) + l1(G(fa), b_))
              + a.lam_id * (l1(G(b_), b_) + l1(Fb(a_), a_)))
        oG.zero_grad(); lg.backward(); oG.step()
        fbp, fap = pB(fb), pA(fa)
        ld = 0
        for D, real, fake in ((DB, b_, fbp), (DA, a_, fap)):
            pr, pf = D(real), D(fake.detach())
            ld = ld + .5 * (mse(pr, torch.ones_like(pr)) + mse(pf, torch.zeros_like(pf)))
        oD.zero_grad(); ld.backward(); oD.step()
        it += 1
        if it % 100 == 0:
            log.append(dict(it=it, sec=round(time.time() - t0), lr=lr, loss_G=float(lg), loss_D=float(ld)))
            print(log[-1], flush=True)
        if it % 250 == 0:
            torch.save(G.state_dict(), os.path.join(OUT, "G_dry2rain.pt"))
    torch.save(G.state_dict(), os.path.join(OUT, "G_dry2rain.pt"))
    torch.save(Fb.state_dict(), os.path.join(OUT, "F_rain2dry.pt"))
    json.dump(dict(level=a.level, lam_id=a.lam_id, iterations=it, minutes=a.minutes, patch=a.size, batch=a.bs, scale=SCALE, log=log,
                   domain_A=len(A), domain_B=len(B)), open(os.path.join(OUT, "train_log.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- translate
@torch.no_grad()
def translate_full(G, img):
    """Translate at half resolution, then add back the original high-frequency detail."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * SCALE) // 4 * 4, int(h * SCALE) // 4 * 4), interpolation=cv2.INTER_AREA)
    out_small = from_t(G(to_t(small)))
    up = cv2.resize(out_small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    detail = img.astype(np.float32) - cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return np.clip(up + detail, 0, 255).astype(np.uint8)


@torch.no_grad()
def translate_crops(G, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        t = torch.cat([to_t(x) for x in X[i:i + bs]])
        out += [from_t(o.unsqueeze(0)) for o in G(t)]
    return np.stack(out)


@torch.no_grad()
def translate_frame_direct(G, img):
    """Crop-level generator applied to a whole frame at native resolution (fully convolutional)."""
    h, w = img.shape[:2]
    return cv2.resize(from_t(G(to_t(cv2.resize(img, (w // 4 * 4, h // 4 * 4))))), (w, h))


def translate(a):
    if a.level == "crop":
        return translate_crop_level(a)
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    G = Gen(); G.load_state_dict(torch.load(os.path.join(OUT, "G_dry2rain.pt"))); G.eval()
    m = manifest()
    spots = collections.defaultdict(list)
    for l in open("data/spots.jsonl"):
        s = json.loads(l); spots[s["image_id"]].append(s)
    if a.split == "train":
        rows = [r for r in m if r["site"] == "UFPR04" and r["split"] == "train"]
        dest = "data/gen_cyclegan_rain.npz"
    else:  # stress test: dry UFPR04 test days -> synthetic rain
        rows = [r for r in m if r["site"] == "UFPR04" and r["split"] == "test" and r["weather"] != "Rainy"]
        dest = "data/stress_cyclegan_rain.npz"
    gal = os.path.join(ROOT, "results", "gallery"); os.makedirs(gal, exist_ok=True)
    X, y, iid = [], [], []
    for k, r in enumerate(rows):
        img = cv2.imread(os.path.join("data/images", r["file"]))
        t = translate_full(G, img)
        if k < 4:
            both = np.hstack([cv2.resize(img, (640, 360)), cv2.resize(t, (640, 360))])
            cv2.imwrite(os.path.join(gal, f"cyclegan_{a.split}_{k}.jpg"), both, [cv2.IMWRITE_JPEG_QUALITY, 85])
        for s in spots[int(r["image_id"])]:
            X.append(rectify(t, s["poly"])); y.append(s["label"]); iid.append(int(r["image_id"]))
    np.savez(dest, X=np.stack(X), y=np.array(y), image_id=np.array(iid))
    print("saved", dest, len(y))


def translate_crop_level(a):
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    G = Gen(); G.load_state_dict(torch.load(os.path.join(OUT, "G_dry2rain.pt"))); G.eval()
    d = np.load("data/crops.npz")
    if a.split == "train":
        m = (d["site"] == "UFPR04") & (d["split"] == "train"); dest = "data/gen_cyclegan_rain.npz"
    else:
        m = (d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] != "Rainy"); dest = "data/stress_cyclegan_rain.npz"
    Xt = translate_crops(G, d["X"][m])
    np.savez(dest, X=Xt, y=d["y"][m], image_id=d["image_id"][m])
    gal = os.path.join(ROOT, "results", "gallery"); os.makedirs(gal, exist_ok=True)
    # gallery 1: crop pairs (top: original dry, middle: CycleGAN rain, bottom: real rain crops for reference)
    rng = np.random.default_rng(1); k = rng.choice(len(Xt), 12, replace=False)
    real = d["X"][(d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] == "Rainy")]
    rk = rng.choice(len(real), 12, replace=False)
    grid = np.vstack([np.hstack(list(d["X"][m][k])), np.hstack(list(Xt[k])), np.hstack(list(real[rk]))])
    cv2.imwrite(os.path.join(gal, f"cyclegan_crops_{a.split}.jpg"), cv2.resize(grid, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    # gallery 2: whole frames translated directly at native resolution
    rows = [r for r in manifest() if r["site"] == "UFPR04" and r["split"] == ("train" if a.split == "train" else "test") and r["weather"] != "Rainy"][:3]
    for j, r in enumerate(rows):
        img = cv2.imread(os.path.join("data/images", r["file"]))
        cv2.imwrite(os.path.join(gal, f"cyclegan_frame_{a.split}_{j}.jpg"),
                    np.hstack([cv2.resize(img, (640, 360)), cv2.resize(translate_frame_direct(G, img), (640, 360))]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("saved", dest, len(Xt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "translate"])
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--split", default="train")
    ap.add_argument("--lam_id", type=float, default=2.0, help="identity-loss weight (cycle weight is 10)")
    ap.add_argument("--level", default="crop", choices=["crop", "frame"],
                    help="crop: CycleGAN on 64x64 bay crops (final); frame: full frames at 1/4 resolution (v2, discarded)")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else translate(a)
