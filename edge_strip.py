"""⏱️ Edge Mode's "strip" timer style (Daryl, 21 Nov 2026 — design E on the overlay canvas).

An extra Timer style beside the ring:

  top     a box snapped to the top edge, centred, only its bottom corners rounded: the 2-digit countdown in a block
          of the phase's colour | the phase's name large. Over the last 3 s a "NEXT ▸ …" tab slides out under it and the
          box flashes as each number comes up. ▸ "ROUND n" once, centre screen, as each round starts.
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
    return {"num": E._font(int(44 * k * S), bold=True), "num_big": E._font(int(50 * k * S), bold=True),
            "label": E._font(int(44 * k * S), bold=True), "label_long": E._font(int(34 * k * S), bold=True),
            "tab_head": E._font(int(12 * k * S), bold=True), "tab_name": E._font(int(19 * k * S), bold=True)}


def _w(d, text, font) -> int:
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


BOX_W, BLOCK_W = 460, 110               # the top box and its colour block (the number), at 720 p
TAB_W, TAB_H = 240, 38                  # ⏭️ the NEXT ▸ tab under the box
TAB_SEC = 0.35                          # how long the tab takes to slide out
WARN_SEC = 3.0                          # ⚠️ the last seconds: tab out, the box flashes as each number comes up


def layout(frame: tuple, *_unused) -> dict:
    """The top box's place and size (fixed for the run): {"w", "h", "x", "k", "block_w", "tab_w", "tab_h"}."""
    W, H = frame
    k = scale(H)
    w = int(BOX_W * k) // 2 * 2
    return {"w": w, "h": max(int(round(BOX_H * k)) // 2 * 2, 16), "x": (W - w) // 2, "k": k,
            "block_w": int(BLOCK_W * k), "tab_w": int(TAB_W * k) // 2 * 2, "tab_h": int(TAB_H * k) // 2 * 2}


def box_image(lay: dict, label: str, ring, block, mode: str = "normal"):
    """The top box (Daryl, 21 Nov — design B): the number's colour block on the left (drawn without the number),
    the phase's name large on the right. mode: "normal" · "edge" (the edge in the phase's colour: the last seconds)
    · "warn" (⚠️ the flash: filled with the phase's colour and glowing round, the block white, the name white)."""
    from PIL import Image, ImageDraw, ImageFilter
    k, w, h = lay["k"], lay["w"], lay["h"]
    f = _fonts(k)
    glow = int(26 * k) if mode == "warn" else 0
    img = Image.new("RGBA", ((w + 2 * glow) * S, (h + glow) * S), (0, 0, 0, 0))
    if glow:                                               # the glow round the box while it flashes
        g = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle((glow * S, -40 * S, (w + glow) * S, h * S), radius=int(18 * k * S),
                                            fill=tuple(ring) + (200,))
        img = Image.alpha_composite(img, g.filter(ImageFilter.GaussianBlur(glow * S / 2)))
    d = ImageDraw.Draw(img)
    ox = glow * S
    rad = int(18 * k * S)
    fill = tuple(ring) + (160,) if mode == "warn" else (10, 10, 12, 189)
    edge = tuple(ring) + (255,) if mode in ("edge", "warn") else (255, 255, 255, 31)
    bw = max(int(2 * k * S), S)
    d.rounded_rectangle((ox, -rad * 2, ox + w * S - 1, h * S - 1), radius=rad, fill=edge)
    d.rounded_rectangle((ox + bw, -rad * 2, ox + w * S - 1 - bw, h * S - 1 - bw), radius=max(rad - bw, 1), fill=fill)
    bwid = lay["block_w"] * S
    d.rounded_rectangle((ox + bw, -rad * 2, ox + bwid, h * S - 1 - bw), radius=max(rad - bw, 1),
                        fill=((255, 255, 255, 255) if mode == "warn" else tuple(block) + (255,)))
    d.rectangle((ox + bwid - rad, -2, ox + bwid, h * S - 1 - bw), fill=((255, 255, 255, 255) if mode == "warn"
                                                                          else tuple(block) + (255,)))
    fl = f["label"] if len(label) <= 7 else f["label_long"]
    room = (w - lay["block_w"]) * S * 0.88
    size = fl.size
    lb = d.textbbox((0, 0), label, font=fl)
    while lb[2] - lb[0] > room and size > 8:
        size = int(size * 0.93)
        fl = E._font(size, bold=True)
        lb = d.textbbox((0, 0), label, font=fl)
    cx = ox + (bwid + w * S) / 2
    d.text((cx - (lb[2] + lb[0]) / 2, h * S / 2 - (lb[3] + lb[1]) / 2), label, font=fl,
           fill=(255, 255, 255, 255) if mode == "warn" else tuple(ring) + (255,))
    return img.resize((w + 2 * glow, h + glow), Image.LANCZOS), glow


def digits_image(lay: dict, text: str, big: bool = False):
    """The number for the colour block (block-wide, box-tall, see-through), in dark type."""
    from PIL import Image, ImageDraw
    k, w, h = lay["k"], lay["block_w"], lay["h"]
    f = _fonts(k)["num_big" if big else "num"]
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), text, font=f)
    d.text((w * S / 2 - (b[2] + b[0]) / 2, h * S / 2 - (b[3] + b[1]) / 2), text, font=f, fill=DARK + (255,))
    return img.resize((w, h), Image.LANCZOS)


def tab_image(lay: dict, next_label: str, colour):
    """⏭️ "NEXT ▸ EXTRA" on a slim tab that slides out from under the box (the name in its own colour)."""
    from PIL import Image, ImageDraw
    k, w, h = lay["k"], lay["tab_w"], lay["tab_h"]
    f = _fonts(k)
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rad = int(14 * k * S)
    bw = max(int(2 * k * S), S)
    d.rounded_rectangle((0, -rad * 2, w * S - 1, h * S - 1), radius=rad, fill=(255, 255, 255, 31))
    d.rounded_rectangle((bw, -rad * 2, w * S - 1 - bw, h * S - 1 - bw), radius=rad, fill=(10, 10, 12, 199))
    a, b = "NEXT ▸", next_label
    ab, bb = d.textbbox((0, 0), a, font=f["tab_head"]), d.textbbox((0, 0), b, font=f["tab_name"])
    gap = 10 * k * S
    tw = (ab[2] - ab[0]) + gap + (bb[2] - bb[0])
    x = (w * S - tw) / 2
    cy = h * S / 2
    d.text((x - ab[0], cy - (ab[3] + ab[1]) / 2), a, font=f["tab_head"], fill=GREY + (255,))
    d.text((x + (ab[2] - ab[0]) + gap - bb[0], cy - (bb[3] + bb[1]) / 2), b, font=f["tab_name"],
           fill=tuple(colour) + (255,))
    return img.resize((w, h), Image.LANCZOS)


ROUND_SEC = 2.4                         # ▸ the round card: on screen this long, once as each round starts


def round_card(r: int, colour, frame: tuple):
    """▸ "ROUND" between two thin lines, the number large under it, a short bar in Wait's colour — centre screen."""
    from PIL import Image, ImageDraw, ImageFilter
    W, H = frame
    k = scale(H)
    w, h = int(420 * k), int(260 * k)
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fh, fn = E._font(int(22 * k * S), bold=True), E._font(int(170 * k * S), bold=True)
    head = "R O U N D"
    hb = d.textbbox((0, 0), head, font=fh)
    cy = int(18 * k * S)
    d.text((w * S / 2 - (hb[2] + hb[0]) / 2, cy - (hb[3] + hb[1]) / 2), head, font=fh, fill=(236, 236, 241, 255))
    lw = int(90 * k * S)
    for x0 in (w * S / 2 - (hb[2] - hb[0]) / 2 - 18 * k * S - lw, w * S / 2 + (hb[2] - hb[0]) / 2 + 18 * k * S):
        d.rectangle((x0, cy - S, x0 + lw, cy + S), fill=(255, 255, 255, 140))
    num = str(int(r))
    nb = d.textbbox((0, 0), num, font=fn)
    ny = int(40 * k * S) - nb[1]
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).text((w * S / 2 - (nb[2] + nb[0]) / 2, ny + 6 * k * S), num, font=fn, fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(14 * k * S)))
    d = ImageDraw.Draw(img)
    d.text((w * S / 2 - (nb[2] + nb[0]) / 2, ny), num, font=fn, fill=(255, 255, 255, 255))
    by = ny + nb[3] + int(14 * k * S)
    d.rounded_rectangle((w * S / 2 - 60 * k * S, by, w * S / 2 + 60 * k * S, by + 6 * k * S), radius=int(3 * k * S),
                        fill=tuple(colour) + (255,))
    return img.resize((w, h), Image.LANCZOS)


def round_alpha(t: float, first: bool) -> float:
    """How strong the round card is t seconds after its round starts (after the box's drop; 0.3 s on the first)."""
    x = t - (0.3 if first else 1.3)
    if x < 0 or x >= ROUND_SEC:
        return 0.0
    return min(x / 0.25, 1.0) if x < 0.25 else (ROUND_SEC - x) / 0.4 if x > ROUND_SEC - 0.4 else 1.0


def tag_image(text: str, bg, fg, frame_h: int, lock: bool = False):
    """✋ / ⏩ a pill above the stroke bar: "🔒 HOLD 3" (hands off), "+5 S" (override)."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    f = E._font(int(16 * k * S), bold=True)
    d0 = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    tw = _w(d0, text, f)
    lk = int(16 * k * S) if lock else 0
    pad, gap = int(14 * k * S), int(8 * k * S) if lock else 0
    w, h = tw + lk + gap + 2 * pad, int(32 * k * S)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=tuple(bg))
    if lock:                                               # a small padlock
        x0, y0 = pad, (h - lk) / 2
        sw = max(int(2.4 * k * S), 2)
        d.rounded_rectangle((x0, y0 + lk * 0.42, x0 + lk, y0 + lk), radius=int(lk * 0.12), outline=tuple(fg), width=sw)
        d.arc((x0 + lk * 0.2, y0, x0 + lk * 0.8, y0 + lk * 0.75), 180, 360, fill=tuple(fg), width=sw)
    b = d.textbbox((0, 0), text, font=f)
    d.text((pad + lk + gap - b[0], h / 2 - (b[3] + b[1]) / 2), text, font=f, fill=tuple(fg))
    return img.resize((max(w // S, 2), max(h // S, 2)), Image.LANCZOS)


def hold_label(frame_h: int):
    """✋ "HOLD" over the marker while it holds at the end of the stroke."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    f = E._font(int(11 * k * S), bold=True)
    d0 = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    w, h = _w(d0, "HOLD", f) + 4 * S, int(14 * k * S)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 * S, 0), "HOLD", font=f, fill=(255, 255, 255, 255))
    return img.resize((max(w // S, 2), max(h // S, 2)), Image.LANCZOS)


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


def bar_marker(frame_h: int, colour, dim: bool = False, ring: bool = False):
    """The stroke meter's marker: an upright pill in the phase's colour (grey and faint: no beat)."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    w, h = max(int(14 * k) // 2 * 2, 4), max(int(22 * k) // 2 * 2, 6)
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    fill = (150, 150, 155, 110) if dim else tuple(colour) + (255,)
    if ring:                                               # ✋ holding at the end: a white ring round it
        pad = int(4 * k * S)
        big = Image.new("RGBA", (w * S + 2 * pad, h * S + 2 * pad), (0, 0, 0, 0))
        dd = ImageDraw.Draw(big)
        dd.rounded_rectangle((0, 0, big.size[0] - 1, big.size[1] - 1), radius=big.size[0] // 2, fill=(255, 255, 255, 255))
        dd.rounded_rectangle((pad, pad, pad + w * S - 1, pad + h * S - 1), radius=w * S // 2, fill=fill)
        return big.resize((big.size[0] // S, big.size[1] // S), Image.LANCZOS)
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


def bar_segment(width: int, frame_h: int, fill=(255, 255, 255, 150)):
    """🎯 The part of the stroke track an instruction allows: the track drawn brighter over that stretch."""
    from PIL import Image, ImageDraw
    k = scale(frame_h)
    th = max(int(6 * k), 2)
    img = Image.new("RGBA", (width * S, th * S), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, width * S - 1, th * S - 1), radius=th * S // 2,
                                          fill=tuple(fill))
    return img.resize((width, th), Image.LANCZOS)


# ---------------------------------------------------------------------------
# 🔔 The phase change (Daryl, 21 Nov: "too subtle" → options 3 + 4): the box drops down with the new phase in it while
# a thick frame in its colour pulses round the picture
# ---------------------------------------------------------------------------

CHANGE_SEC = 1.45                       # how long the change shows
DROP_IN, DROP_HOLD, DROP_OUT = 0.2, 1.05, 1.3     # the box: down by, held until, back up by (seconds)
BORDER = 18                             # the frame's width at 720 p (plus a soft glow inside it)


DROP_SUB = {"HANDS OFF": "LET GO · WAIT FOR IT", "OVERRIDE": "DON'T STOP · KEEP THE PACE", "NOT YET": "KEEP GOING"}


def drop_panel(label: str, colour, frame: tuple, head: str = "", sub: str = None):
    """The dropped-down box: the phase's colour and its name huge in dark text, bottom corners round;
    a surprise adds a small line saying what to do (DROP_SUB — Daryl, 21 Nov)."""
    from PIL import Image, ImageDraw
    W, H = frame
    k = scale(H)
    w, h = int(560 * k) // 2 * 2, int(200 * k) // 2 * 2
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rad = int(28 * k * S)
    d.rounded_rectangle((0, -rad * 2, w * S - 1, h * S - 1), radius=rad, fill=tuple(colour) + (255,))
    fn = E._font(int(16 * k * S), bold=True)
    size = int(110 * k * S)
    fl = E._font(size, bold=True)
    lb = d.textbbox((0, 0), label, font=fl)
    while lb[2] - lb[0] > w * S * 0.9 and size > 8:
        size = int(size * 0.93)
        fl = E._font(size, bold=True)
        lb = d.textbbox((0, 0), label, font=fl)
    sub = DROP_SUB.get(label) if sub is None else sub
    nb = d.textbbox((0, 0), head, font=fn) if head else (0, 0, 0, 0)
    fs = E._font(int(36 * k * S), bold=True)                 # (21 Nov: twice as big; "NOW" dropped)
    sb = d.textbbox((0, 0), sub, font=fs) if sub else (0, 0, 0, 0)
    while sub and sb[2] - sb[0] > w * S * 0.9 and fs.size > 8:
        fs = E._font(int(fs.size * 0.94), bold=True)
        sb = d.textbbox((0, 0), sub, font=fs)
    g = 10 * k * S
    total = ((nb[3] - nb[1]) + g if head else 0) + (lb[3] - lb[1]) + ((sb[3] - sb[1]) + g if sub else 0)
    top = (h * S - total) / 2
    if head:
        d.text((w * S / 2 - (nb[2] + nb[0]) / 2, top - nb[1]), head, font=fn, fill=DARK + (255,))
    ly = top + ((nb[3] - nb[1]) + g if head else 0)
    d.text((w * S / 2 - (lb[2] + lb[0]) / 2, ly - lb[1]), label, font=fl, fill=DARK + (255,))
    if sub:
        d.text((w * S / 2 - (sb[2] + sb[0]) / 2, ly + (lb[3] - lb[1]) + g - sb[1]), sub, font=fs, fill=DARK + (230,))
    return img.resize((w, h), Image.LANCZOS)


BACK_ON_SEC = 0.65                      # ✋ the "BACK ON" drop as a hands off ends: about half as long


def back_on_share(t: float) -> float:
    return drop_share(t * 2.0)


def drop_share(t: float) -> float:
    """How much of the dropped box shows t seconds into the phase (0 … 1)."""
    if t < 0 or t >= DROP_OUT:
        return 0.0
    if t < DROP_IN:
        x = t / DROP_IN
        return 1 - (1 - x) ** 3
    if t <= DROP_HOLD:
        return 1.0
    x = (t - DROP_HOLD) / (DROP_OUT - DROP_HOLD)
    return 1 - x * x


def border_level(t: float) -> float:
    """The frame's strength t seconds into the phase: three pulses, then gone by CHANGE_SEC."""
    marks = [(0.0, 1.0), (0.24, 0.15), (0.48, 1.0), (0.72, 0.15), (0.96, 1.0), (CHANGE_SEC, 0.0)]
    if t < 0 or t >= CHANGE_SEC:
        return 0.0
    for (t0, a0), (t1, a1) in zip(marks, marks[1:]):
        if t0 <= t <= t1:
            return a0 + (a1 - a0) * (t - t0) / (t1 - t0)
    return 0.0


def border_strips(colour, frame: tuple) -> list:
    """The frame as four edge pictures [(picture, x, y)]: a solid band in the phase's colour fading inwards."""
    import numpy as np
    from PIL import Image
    W, H = frame
    k = scale(H)
    solid, glow = max(int(BORDER * k), 3), max(int(60 * k), 6)
    depth = solid + glow
    ramp = np.concatenate([np.full(solid, 255.0), np.linspace(140.0, 0.0, glow)]).astype(np.uint8)
    out = []
    for side in ("top", "bottom", "left", "right"):
        if side in ("top", "bottom"):
            a = np.repeat((ramp if side == "top" else ramp[::-1])[:, None], W, axis=1)
            x, y = 0, 0 if side == "top" else H - depth
        else:
            a = np.repeat((ramp if side == "left" else ramp[::-1])[None, :], H, axis=0)
            x, y = (0 if side == "left" else W - depth), 0
        rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = tuple(colour)
        rgba[..., 3] = a
        out.append((Image.fromarray(rgba, "RGBA"), x, y))
    return out
