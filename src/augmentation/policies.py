"""Augmentation 4 ระดับตาม Experiment B — ทำงานบน GPU ทีละ batch

    none              ไม่ทำอะไร (baseline)
    standard          rotate / translate / scale เล็กน้อย
    thai_specific     ออกแบบจาก failure mode ของภาพตัวอักษรไทย:
                      rotation, perspective, blur, noise, uneven lighting, stroke variation
    thai_randaugment  thai_specific + สุ่มหยิบ op เพิ่มแบบ RandAugment

input/output ของทุก policy คือ float tensor (B, 1, H, W) ค่าอยู่ใน [0, 1] พื้นขาว = 1
ข้อควรระวัง: ห้าม flip และห้ามหมุนแรง เพราะตัวอักษรไทยหลายคู่ต่างกันเพียงรายละเอียดเล็ก
(ผ/ฝ, ช/ซ, ฎ/ฏ, ี/ื) การบิดแรงเกินทำให้ label ไม่ตรงกับภาพ
"""
import torch
import torch.nn.functional as F

POLICIES = ("none", "standard", "thai_specific", "thai_randaugment")


def _rand(b, lo, hi, device):
    return torch.empty(b, device=device).uniform_(lo, hi)


def _warp(x, theta):
    """apply affine/perspective บน "หมึก" (1-x) เพื่อให้ขอบที่ล้นออกไปเป็นพื้นขาว"""
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return 1 - F.grid_sample(1 - x, grid, padding_mode="zeros", align_corners=False)


def affine(x, degrees=10, translate=0.08, scale=(0.85, 1.1), shear=8):
    b, d = x.size(0), x.device
    ang = torch.deg2rad(_rand(b, -degrees, degrees, d))
    sh = torch.deg2rad(_rand(b, -shear, shear, d))
    sc = _rand(b, *scale, d)
    cos, sin, tsh = torch.cos(ang), torch.sin(ang), torch.tan(sh)
    t = 2 * translate
    theta = torch.stack([
        torch.stack([cos / sc, (-sin + cos * tsh) / sc, _rand(b, -t, t, d)], 1),
        torch.stack([sin / sc, (cos + sin * tsh) / sc, _rand(b, -t, t, d)], 1),
    ], 1)
    return _warp(x, theta)


def perspective(x, strength=0.12):
    """จำลองการถ่ายเอกสารจากมุมเอียง โดยรบกวนเมทริกซ์ affine เล็กน้อย"""
    b, d = x.size(0), x.device
    base = torch.eye(2, 3, device=d).expand(b, 2, 3).clone()
    return _warp(x, base + torch.randn(b, 2, 3, device=d) * strength)


def blur(x, p=0.5, max_sigma=1.2):
    """จำลองภาพจากกล้อง/สแกนเนอร์ด้วย gaussian blur เบา ๆ"""
    b, d = x.size(0), x.device
    sigma = float(torch.empty(1).uniform_(0.3, max_sigma))  # หนึ่งค่าต่อ batch พอ
    k = torch.arange(-2, 3, device=d).float()
    w = torch.exp(-k ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    out = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), w.view(1, 1, 1, 5))
    out = F.conv2d(out, w.view(1, 1, 5, 1))
    use = (torch.rand(b, 1, 1, 1, device=d) < p).float()
    return out * use + x * (1 - use)


def noise(x, p=0.5, sigma=0.06):
    """จำลองภาพคุณภาพต่ำ"""
    use = (torch.rand(x.size(0), 1, 1, 1, device=x.device) < p).float()
    return (x + torch.randn_like(x) * sigma * use).clamp(0, 1)


def uneven_lighting(x, p=0.5, strength=0.25):
    """จำลองแสงไม่สม่ำเสมอด้วย gradient คูณทั้งภาพ"""
    b, _, h, w = x.shape
    d = x.device
    gy = torch.linspace(-1, 1, h, device=d)[None, :, None]
    gx = torch.linspace(-1, 1, w, device=d)[None, None, :]
    a, c = _rand(b, -1, 1, d)[:, None, None], _rand(b, -1, 1, d)[:, None, None]
    ramp = 1 + strength * (a * gy + c * gx)
    use = (torch.rand(b, 1, 1, device=d) < p).float()
    return (x * (ramp * use + (1 - use))[:, None]).clamp(0, 1)


def stroke_variation(x, p=0.5):
    """จำลองปากกาหนา/บาง ด้วย morphological dilate-erode บนหมึก"""
    b, d = x.size(0), x.device
    ink = 1 - x
    thick = F.max_pool2d(ink, 3, 1, 1)                     # เส้นหนาขึ้น
    thin = -F.max_pool2d(-ink, 3, 1, 1)                    # เส้นบางลง
    r = torch.rand(b, 1, 1, 1, device=d)
    out = torch.where(r < p / 2, thick, torch.where(r < p, thin, ink))
    return 1 - out


def policy_none(x):
    return x


def policy_standard(x):
    return affine(x, degrees=10, translate=0.08, scale=(0.9, 1.1), shear=0)


def policy_thai(x):
    x = affine(x, degrees=10, translate=0.08, scale=(0.85, 1.1), shear=8)
    x = perspective(x, 0.06)
    x = stroke_variation(x, p=0.4)
    x = blur(x, p=0.3)
    x = uneven_lighting(x, p=0.3)
    return noise(x, p=0.3)


def policy_thai_randaugment(x, n_ops=2):
    """thai_specific แล้วสุ่มหยิบ op เพิ่มอีก n_ops ตัวแบบ RandAugment"""
    x = policy_thai(x)
    ops = [lambda t: affine(t, 8, 0.05, (0.9, 1.1), 5),
           lambda t: perspective(t, 0.08),
           lambda t: blur(t, p=1.0),
           lambda t: noise(t, p=1.0, sigma=0.08),
           lambda t: uneven_lighting(t, p=1.0, strength=0.35),
           lambda t: stroke_variation(t, p=1.0)]
    for i in torch.randperm(len(ops))[:n_ops].tolist():
        x = ops[i](x)
    return x


_POLICY_FN = {"none": policy_none, "standard": policy_standard,
              "thai_specific": policy_thai, "thai_randaugment": policy_thai_randaugment}


def get_policy(name):
    if name not in _POLICY_FN:
        raise ValueError(f"ไม่รู้จัก augmentation: {name} (มีให้เลือก {POLICIES})")
    return _POLICY_FN[name]


def to_input(x_uint8, device, policy=None):
    """uint8 (B,H,W) -> float (B,3,H,W) normalized พร้อม augmentation (ถ้าส่ง policy มา)"""
    x = x_uint8.to(device).float().div_(255).unsqueeze(1)
    if policy is not None:
        x = policy(x)
    return ((x - 0.5) / 0.5).expand(-1, 3, -1, -1)
