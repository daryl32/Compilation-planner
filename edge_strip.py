"""⏱️ Edge Mode's "strip" timer style (Daryl, 21 Nov 2026 — design E on the overlay canvas).

An extra Timer style beside the ring:

  top     a box snapped to the top edge, centred, only its bottom corners rounded:
          2-digit countdown | the current phase (large, in its colour) with a small grey "NEXT ▸ …" under it | ROUND 2/4
  bottom  a thin full-width strip holding only the horizontal stroke meter — the marker slides left ↔ right on the beats

🔔 signals: as a phase starts the box fills with the phase's colour (dark text) for a moment; over the last PULSE_SEC
seconds the number grows and takes the phase's colour and the box's edge lights up.

Sizes follow the mockup at 720 p and scale with the frame's height. Pictures are drawn with PIL at 3× and shrunk
(smooth edges), as edge_mode's ring is; edge_render lays them on the frames.
"""

from __future__ import annotations

import math

import edge_mode as E

S = 3                                   # drawn this many times larger, then shrunk
BOX_H = 62                              # the top box's height at 720 p
BAR_H = 34                              # the bottom strip's height at 720 p
BAND_H = 60                             # 🗒️ with instructions (◧ split screen): the strip is this tall
FLASH_SEC = 0.5                         # 🔔 how long the box stays filled with the new phase's colour
DARK = (17, 17, 20)
GREY = (142, 142, 152)


def scale(frame_h: int) -> float:
    return float(frame_h) / 720.0


def bar_height(frame_h: int) -> int:
    return max(int(round(BAR_H * scale(frame_h))) // 2 * 2, 8)


def _fonts(k: float) -> dict:
    return {"num": E._font(int(44 * k * S), bold=True), "num_pulse": E._font(int(50 * k * S), bold=True),
            "label": E._font(int(26 * k * S), bold=True), "next": E._font(int(11 * k * S), bold=True),
            "round": E._font(int(14 * k * S), bold=True)}


def _w(d, text, font) -> int:
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def layout(frame: tuple, labels, nexts, rounds: int) -> dict:
    """The box's fixed layout (it never changes width mid-run): {"w", "h", "x", "slot" (digit area: x, w), "mid_x",
    "mid_w", "round_x"} in final pixels, sized for the widest label / next / round in use."""
    from PIL import Image, ImageDraw
    W, H = frame
    k = scale(H)
    f = _fonts(k)
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    pad, gap, div = 28 * k, 20 * k, 1 * k
    slot = max(_w(d, t, f["num_pulse"]) for t in ("88", "+15")) / S + 6 * k
    mid = max([_w(d, lab, f["label"]) for lab in labels] + [_w(d, f"NEXT ▸ {n}", f["next"]) for n in nexts]) / S
    rnd = _w(d, f"ROUND {rounds}/{rounds}", f["round"]) / S
    w = int(math.ceil(pad + slot + gap + div + gap + mid + gap + div + gap + rnd + pad)) // 2 * 2
    h = max(int(round(BOX_H * k)) // 2 * 2, 16)
    x = (W - w) // 2
    return {"w": w, "h": h, "x": x, "k": k, "slot_x": int(pad), "slot_w": int(math.ceil(slot)),
            "div1": int(pad + slot + gap), "mid_x": int(pad + slot + gap + div + gap), "mid_w": int(math.ceil(mid)),
            "div2": int(pad + slot + gap + div + gap + mid + gap), "round_x": int(pad + slot + gap + div + gap + mid + gap + div + gap)}


def box_image(lay: dict, label: str, colour, next_label: str, r: int, n: int, mode: str = "normal"):
    """The top box for one stretch, without the number (drawn in its slot every second). mode: "normal", "flash"
    (filled with the phase's colour, dark text) or "pulse" (its edge in the phase's colour)."""
    from PIL import Image, ImageDraw
    k, w, h = lay["k"], lay["w"], lay["h"]
    f = _fonts(k)
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rad = int(18 * k * S)
    fill = tuple(colour) + (255,) if mode == "flash" else (10, 10, 12, 184)
    edge = tuple(colour) + (255,) if mode in ("flash", "pulse") else (255, 255, 255, 31)
    bw = max(int(2 * k * S), S)
    # only the bottom corners rounded: a rounded box drawn taller than the picture, its top cut off
    d.rounded_rectangle((0, -rad * 2, w * S - 1, h * S - 1), radius=rad, fill=edge)
    d.rounded_rectangle((bw, -rad * 2, w * S - 1 - bw, h * S - 1 - bw), radius=max(rad - bw, 1), fill=fill)
    text = DARK + (255,) if mode == "flash" else None
    cy = h * S / 2
    dv = (DARK + (110,)) if mode == "flash" else (255, 255, 255, 56)
    for xx in (lay["div1"], lay["div2"]):
        d.rectangle((xx * S, cy - 17 * k * S, xx * S + max(int(k * S), 1), cy + 17 * k * S), fill=dv)
    lb = d.textbbox((0, 0), label, font=f["label"])
    nb = d.textbbox((0, 0), "NEXT ▸ " + next_label, font=f["next"])
    lh, nh, g = lb[3] - lb[1], nb[3] - nb[1], 5 * k * S
    top = cy - (lh + g + nh) / 2
    d.text((lay["mid_x"] * S - lb[0], top - lb[1]), label, font=f["label"], fill=text or (tuple(colour) + (255,)))
    d.text((lay["mid_x"] * S - nb[0], top + lh + g - nb[1]), "NEXT ▸ " + next_label, font=f["next"],
           fill=text or (GREY + (255,)))
    rt = f"ROUND {int(r)}/{int(n)}"
    rb = d.textbbox((0, 0), rt, font=f["round"])
    d.text((lay["round_x"] * S - rb[0], cy - (rb[3] + rb[1]) / 2), rt, font=f["round"], fill=text or (228, 228, 234, 255))
    return img.resize((w, h), Image.LANCZOS)


def digits_image(lay: dict, text: str, colour=(255, 255, 255), pulse: bool = False):
    """The number for the box's slot (slot-wide, box-tall, see-through)."""
    from PIL import Image, ImageDraw
    k, w, h = lay["k"], lay["slot_w"], lay["h"]
    f = _fonts(k)["num_pulse" if pulse else "num"]
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), text, font=f)
    d.text((w * S / 2 - (b[2] + b[0]) / 2, h * S / 2 - (b[3] + b[1]) / 2), text, font=f, fill=tuple(colour) + (255,))
    return img.resize((w, h), Image.LANCZOS)


def number_text(look: str, left: float) -> str:
    secs = int(math.ceil(left - 1e-6))
    return f"+{secs:d}" if look in ("override", "notyet") else f"{min(secs, 99):02d}"


def band_height(frame_h: int) -> int:
    return max(int(round(BAND_H * scale(frame_h))) // 2 * 2, 12)


def bar_track(width: int, frame_h: int, alpha: float = 0.45, height: int = None):
    """The bottom strip: a dark see-through band `width` wide with a thin light track along its middle. Returns
    (picture, track x0, track x1) — the track's ends in the picture. height: the band's (default: the strip's)."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    h = int(height or bar_height(frame_h))
    img = Image.new("RGBA", (width * S, h * S), (0, 0, 0, int(255 * alpha)))
    d = ImageDraw.Draw(img)
    pad = int(40 * k)
    th = max(int(6 * k * S), S)
    d.rounded_rectangle((pad * S, h * S / 2 - th / 2, (width - pad) * S, h * S / 2 + th / 2), radius=th // 2,
                        fill=(255, 255, 255, 46))
    return img.resize((width, h), Image.LANCZOS), pad, width - pad


def bar_marker(frame_h: int, colour, dim: bool = False):
    """The stroke meter's marker: an upright pill in the phase's colour (grey and faint: no beat)."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    w, h = max(int(14 * k) // 2 * 2, 4), max(int(22 * k) // 2 * 2, 6)
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    fill = (150, 150, 155, 110) if dim else tuple(colour) + (255,)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * S - 1, h * S - 1), radius=w * S // 2, fill=fill)
    return img.resize((w, h), Image.LANCZOS)


def instruction_panel(text: str, colour, width: int, frame_h: int, alpha: float = 0.6):
    """🗒️ Her half of the bottom band: the instruction in large white capitals beside a short bar in the phase's
    colour, on a dark see-through band with a thin divider on its left edge. Long text shrinks to fit."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    h = band_height(frame_h)
    img = Image.new("RGBA", (width * S, h * S), (0, 0, 0, int(255 * alpha)))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, max(int(k * S), 1), h * S), fill=(255, 255, 255, 36))
    text = text.upper()
    size = int(30 * k * S)
    f = E._font(size, bold=True)
    acc_w, gap = 6 * k * S, 12 * k * S
    room = width * S - 2 * 24 * k * S - acc_w - gap
    while _w(d, text, f) > room and size > 8:
        size = int(size * 0.92)
        f = E._font(size, bold=True)
    b = d.textbbox((0, 0), text, font=f)
    tw = b[2] - b[0]
    x = (width * S - (acc_w + gap + tw)) / 2
    cy = h * S / 2
    ah = 30 * k * S
    if text:                                               # (an empty band before the first instruction)
        d.rounded_rectangle((x, cy - ah / 2, x + acc_w, cy + ah / 2), radius=acc_w / 2, fill=tuple(colour) + (255,))
        d.text((x + acc_w + gap - b[0], cy - (b[3] + b[1]) / 2), text, font=f, fill=(255, 255, 255, 255))
    return img.resize((width, h), Image.LANCZOS)
