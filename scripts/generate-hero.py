#!/usr/bin/env python3
# BisQue Ultra GitHub hero generator.
#   deps: pip install numpy pillow   ·   rasterize: rsvg-convert (librsvg)
#   run:  python3 scripts/generate-hero.py
#         rsvg-convert -w 1280 -h 640 .github/assets/bisque-ultra-hero.svg -o .github/assets/bisque-ultra-hero.png
# Tweak the tagline / BANDS / colours below, then re-run.
"""Generate a BisQue Ultra GitHub hero: an app-window mockup of the Lens view
featuring the CIFTI carpet plot. Emits a self-contained SVG (embedded carpet
raster + vector chrome). Rasterize with rsvg-convert."""
import base64, io, html
import numpy as np
from PIL import Image

import os
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "assets")

# ---------------------------------------------------------------- carpet raster
def rdbu_lut():
    stops = [(5,48,97),(33,102,172),(67,147,195),(146,197,222),(209,229,240),
             (247,247,247),(253,219,199),(244,165,130),(214,96,77),(178,24,43),(103,0,31)]
    xs = np.linspace(0, 1, len(stops)); lut = np.zeros((256, 3))
    for c in range(3):
        lut[:, c] = np.interp(np.linspace(0, 1, 256), xs, [s[c] for s in stops])
    return lut.astype(np.uint8)

BANDS = [("Cortex L",30),("Cortex R",30),("Cerebellum L",9),("Cerebellum R",9),
         ("Brain Stem",4),("Thalamus",3),("Hippocampus",2),("Putamen",2),
         ("Caudate",2),("Amygdala",1)]
TW = sum(w for _, w in BANDS)
RW, RH = 1200, 520                    # raster time x grayordinate resolution
SEP = 2                                # white separator px between structures
rng = np.random.default_rng(7)

# global signal events (vertical stripes) — like real BOLD global fluctuations
gsig = np.zeros(RW)
for _ in range(7):
    t0 = rng.integers(0, RW); w = rng.integers(5, 16); gsig[max(0,t0-w):t0+w] += rng.normal(0, 1.3)
gsig = np.convolve(gsig, np.ones(9)/9, mode="same")

rows_rgb = []
lut = rdbu_lut()
band_spans = []   # (name, start_row, end_row) in final raster space
cur = 0
for name, w in BANDS:
    n = max(3, round(w / TW * (RH - SEP*len(BANDS))))
    ssig = np.convolve(rng.normal(0, 1, RW), np.ones(17)/17, mode="same")  # structure-shared signal
    block = np.empty((n, RW))
    for r in range(n):
        row = 0.55*gsig + 0.6*ssig + rng.normal(0, 1, RW)
        block[r] = np.convolve(row, np.ones(3)/3, mode="same")
    block = (block - block.mean(1, keepdims=True)) / (block.std(1, keepdims=True) + 1e-6)
    block = np.clip(block, -3, 3)
    idx = ((block + 3) / 6 * 255).astype(np.uint8)
    band_spans.append((name, cur, cur + n)); cur += n
    rows_rgb.append(lut[idx])
    rows_rgb.append(np.full((SEP, RW, 3), 250, np.uint8))  # separator
raster = np.concatenate(rows_rgb, 0)[:RH]
buf = io.BytesIO(); Image.fromarray(raster, "RGB").save(buf, "PNG")
carpet_b64 = base64.b64encode(buf.getvalue()).decode()
RASTER_H = raster.shape[0]

# ------------------------------------------------------------------- SVG layout
W, H = 1280, 640
# window
wx, wy, ww, wh = 40, 120, 1200, 480
tb = 34                                     # title bar height
sbx, sbw = wx, 148                           # sidebar
mx = sbx + sbw                                # main panel left
# plot area inside main
plot_x, plot_y = mx + 118, wy + tb + 78
plot_r = wx + ww - 84                        # leave room for colourbar
plot_b = wy + wh - 40
plot_w, plot_h = plot_r - plot_x, plot_b - plot_y

def esc(s): return html.escape(str(s))

# structure label positions (declutter: min spacing, mirrors the real viewer)
LABEL_MIN = 20
centers = [plot_y + (s + e) / 2 / RASTER_H * plot_h for _, s, e in band_spans]
ly = centers[:]
for i in range(1, len(ly)):
    if ly[i] - ly[i-1] < LABEL_MIN: ly[i] = ly[i-1] + LABEL_MIN
if ly and ly[-1] > plot_b:
    ly[-1] = plot_b
    for i in range(len(ly)-2, -1, -1):
        if ly[i+1] - ly[i] < LABEL_MIN: ly[i] = ly[i+1] - LABEL_MIN

nav = [("New chat", False), ("Resources", False), ("Training", False),
       ("Lens", True), ("Admin", False)]

svg = []
def A(s): svg.append(s)

A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">')
A('<defs>')
A('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
  '<stop offset="0" stop-color="#0b0f1c"/><stop offset="1" stop-color="#070a14"/></linearGradient>')
A('<radialGradient id="glow" cx="0.82" cy="0.12" r="0.9">'
  '<stop offset="0" stop-color="#1b3a5b" stop-opacity="0.55"/><stop offset="0.5" stop-color="#12203a" stop-opacity="0.18"/>'
  '<stop offset="1" stop-color="#070a14" stop-opacity="0"/></radialGradient>')
A('<linearGradient id="cbar" x1="0" y1="0" x2="0" y2="1">'
  '<stop offset="0" stop-color="#67001f"/><stop offset="0.5" stop-color="#f7f7f7"/><stop offset="1" stop-color="#053061"/></linearGradient>')
A('<linearGradient id="mark" x1="0" y1="0" x2="1" y2="1">'
  '<stop offset="0" stop-color="#5b9bd5"/><stop offset="1" stop-color="#c0405a"/></linearGradient>')
A(f'<clipPath id="plotclip"><rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="5"/></clipPath>')
A('<clipPath id="winclip"><rect x="%d" y="%d" width="%d" height="%d" rx="14"/></clipPath>' % (wx,wy,ww,wh))
A('<filter id="shadow" x="-20%%" y="-20%%" width="140%%" height="160%%">'
  '<feDropShadow dx="0" dy="18" stdDeviation="34" flood-color="#000000" flood-opacity="0.55"/></filter>')
A('</defs>')

# background
A(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
A(f'<rect width="{W}" height="{H}" fill="url(#glow)"/>')

# ---- brand lockup (top-left) ----
A('<g transform="translate(48,44)">')
A('<circle cx="17" cy="17" r="17" fill="url(#mark)"/>')
A('<ellipse cx="17" cy="17" rx="20" ry="8" fill="none" stroke="#7fb0e0" stroke-width="1.6" opacity="0.7" transform="rotate(-25 17 17)"/>')
A('<circle cx="17" cy="17" r="6.5" fill="#0b0f1c"/><path d="M17 12 l1.6 3.4 3.7 .4 -2.8 2.5 .8 3.6 -3.3-1.9 -3.3 1.9 .8-3.6 -2.8-2.5 3.7-.4z" fill="#ffd76a"/>')
A('<text x="46" y="16" fill="#f4f6fb" font-size="23" font-weight="700" letter-spacing="-0.3">BisQue Ultra</text>')
A('<text x="46" y="37" fill="#93a0bd" font-size="13.5" letter-spacing="0.2">Agentic AI + scientific imaging, in the browser</text>')
A('</g>')

# ---- app window ----
A(f'<rect x="{wx}" y="{wy}" width="{ww}" height="{wh}" rx="14" fill="#0f1523" stroke="#232c40" stroke-width="1" filter="url(#shadow)"/>')
A('<g clip-path="url(#winclip)">')
# title bar
A(f'<rect x="{wx}" y="{wy}" width="{ww}" height="{tb}" fill="#0c1120"/>')
for i, col in enumerate(["#f2564b", "#f5b83d", "#4cc26a"]):
    A(f'<circle cx="{wx+22+i*20}" cy="{wy+tb/2}" r="5.5" fill="{col}"/>')
A(f'<text x="{wx+ww/2}" y="{wy+tb/2+4}" text-anchor="middle" fill="#7c88a3" font-size="12.5">BisQue Ultra  ·  Lens</text>')
# sidebar
A(f'<rect x="{sbx}" y="{wy+tb}" width="{sbw}" height="{wh-tb}" fill="#0b0f1c"/>')
A(f'<line x1="{sbx+sbw}" y1="{wy+tb}" x2="{sbx+sbw}" y2="{wy+wh}" stroke="#1c2436" stroke-width="1"/>')
A(f'<circle cx="{sbx+24}" cy="{wy+tb+26}" r="9" fill="url(#mark)"/>')
A(f'<text x="{sbx+40}" y="{wy+tb+30}" fill="#e6ebf5" font-size="13" font-weight="700">BisQue Ultra</text>')
ny = wy + tb + 64
for label, active in nav:
    if active:
        A(f'<rect x="{sbx+12}" y="{ny-15}" width="{sbw-24}" height="26" rx="7" fill="#1a2740"/>')
    A(f'<circle cx="{sbx+26}" cy="{ny-2}" r="3.2" fill="{"#6ea8e6" if active else "#5b667f"}"/>')
    A(f'<text x="{sbx+40}" y="{ny+2}" fill="{"#eaf0fb" if active else "#8b96af"}" font-size="12.5" font-weight="{600 if active else 400}">{esc(label)}</text>')
    ny += 34
A(f'<text x="{sbx+20}" y="{wy+wh-18}" fill="#5b667f" font-size="10.5">Guest · guest access</text>')

# main: header row
hx = mx + 22
A(f'<text x="{hx}" y="{wy+tb+30}" fill="#eef2fb" font-size="14.5" font-weight="650">rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii</text>')
A(f'<rect x="{hx}" y="{wy+tb+42}" width="52" height="18" rx="9" fill="#161d2e" stroke="#2b3444" stroke-width="1"/>')
A(f'<text x="{hx+26}" y="{wy+tb+55}" text-anchor="middle" fill="#8ea3c6" font-size="10" font-family="monospace">CIFTI-2</text>')
A(f'<text x="{hx+64}" y="{wy+tb+55}" fill="#8892a8" font-size="11.5" font-family="monospace">dense timeseries · 91,282 × 1,200 · TR 0.72s</text>')
# tabs (right)
tabx = plot_r - 6
A(f'<rect x="{tabx-176}" y="{wy+tb+40}" width="176" height="24" rx="7" fill="#10182a"/>')
A(f'<rect x="{tabx-174}" y="{wy+tb+42}" width="92" height="20" rx="6" fill="#e9eefb"/>')
A(f'<text x="{tabx-128}" y="{wy+tb+56}" text-anchor="middle" fill="#0d1424" font-size="11.5" font-weight="600">Carpet plot</text>')
A(f'<text x="{tabx-40}" y="{wy+tb+56}" text-anchor="middle" fill="#8892a8" font-size="11.5">Connectivity</text>')

# carpet image + frame
A(f'<image x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" preserveAspectRatio="none" '
  f'clip-path="url(#plotclip)" href="data:image/png;base64,{carpet_b64}"/>')
A(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" rx="5" fill="none" stroke="#28324a" stroke-width="1"/>')
# structure labels
for (name, _, _), yy in zip(band_spans, ly):
    A(f'<text x="{plot_x-12}" y="{yy+3.5}" text-anchor="end" fill="#c7d0e2" font-size="10.5" font-weight="600">{esc(name)}</text>')
# colourbar
cbx = plot_r + 22
A(f'<rect x="{cbx}" y="{plot_y+30}" width="12" height="180" rx="2" fill="url(#cbar)" stroke="#28324a" stroke-width="0.8"/>')
for t, lab in [(0,"+3σ"),(0.5,"0"),(1.0,"−3σ")]:
    A(f'<text x="{cbx+18}" y="{plot_y+34+t*180}" fill="#8892a8" font-size="10" font-family="monospace">{lab}</text>')
# frame axis
for i in range(0, 7):
    fx = plot_x + plot_w * i/6
    A(f'<line x1="{fx}" y1="{plot_b}" x2="{fx}" y2="{plot_b+4}" stroke="#3a4760" stroke-width="1"/>')
    A(f'<text x="{fx}" y="{plot_b+16}" text-anchor="middle" fill="#758198" font-size="9.5" font-family="monospace">{i*200}</text>')
A(f'<text x="{plot_x+plot_w/2}" y="{plot_b+30}" text-anchor="middle" fill="#5b667f" font-size="10">frame</text>')
A('</g>')  # winclip

A('</svg>')
svg_str = "\n".join(svg)
open(os.path.join(OUT, "bisque-ultra-hero.svg"), "w").write(svg_str)
print("wrote SVG:", len(svg_str), "bytes; raster", RW, "x", RASTER_H, "; bands", len(band_spans))
