"""Generate the plugin listing icon: a stylised molecular graph.

Square PNG, 1024x1024, flat colours, no external assets. Regenerate with:
    python examples/make_icon.py
Writes .claude-plugin/icon.png
"""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), ".claude-plugin", "icon.png")

S = 1024          # final size
SS = 4            # supersample factor for smooth edges
PAD = 0.12        # fraction of the canvas kept clear on every side

BG = (22, 27, 34)
BOND = (122, 142, 162)
NODE_C = (233, 236, 239)
NODE_N = (104, 158, 212)
NODE_O = (209, 110, 100)


def hexagon(cx: float, cy: float, r: float, rot: float = 0.0):
    return [
        (cx + r * math.cos(rot + i * math.pi / 3.0),
         cy + r * math.sin(rot + i * math.pi / 3.0))
        for i in range(6)
    ]


def build(size: float):
    """Fused bicycle, upright, with three well-separated substituents."""
    cx, cy = size / 2, size / 2
    r = size * 0.175

    # Rings stacked vertically so the whole glyph is nearly square.
    dy = r * math.sqrt(3) / 2
    top = hexagon(cx, cy - dy, r, rot=0.0)
    bot = hexagon(cx, cy + dy, r, rot=0.0)

    bonds, nodes = [], []

    def add_ring(ring):
        for i in range(6):
            bonds.append((ring[i], ring[(i + 1) % 6]))
        nodes.extend(ring)

    add_ring(top)
    add_ring(bot)

    # Substituents hang off free vertices only, pointing directly away from
    # their ring centre so they can never cross the scaffold. Vertices 1 and 2
    # of the top ring and 4 and 5 of the bottom ring form the shared (fusion)
    # edge, so those are excluded.
    subs = []
    L = size * 0.13
    top_c = (cx, cy - dy)
    bot_c = (cx, cy + dy)
    for ring, centre, idx, col in (
        (top, top_c, 4, NODE_O),   # upper left
        (top, top_c, 5, NODE_N),   # upper right
        (bot, bot_c, 1, NODE_N),   # lower right
    ):
        v = ring[idx]
        ux, uy = v[0] - centre[0], v[1] - centre[1]
        n = math.hypot(ux, uy) or 1.0
        end = (v[0] + L * ux / n, v[1] + L * uy / n)
        bonds.append((v, end))
        subs.append((end, col))

    return bonds, nodes, subs, (top, bot)


def main() -> None:
    size = S * SS
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)

    bonds, nodes, subs, rings = build(size)

    # The glyph is roughly twice as tall as it is wide. Rotating it 45 degrees
    # makes its bounding box square, so it fills the icon instead of leaving
    # wide margins either side.
    def rot(p, ang=math.pi / 4, c=(size / 2, size / 2)):
        dx, dy = p[0] - c[0], p[1] - c[1]
        ca, sa = math.cos(ang), math.sin(ang)
        return (c[0] + dx * ca - dy * sa, c[1] + dx * sa + dy * ca)

    bonds = [(rot(a), rot(b)) for a, b in bonds]
    nodes = [rot(p) for p in nodes]
    subs = [(rot(p), col) for p, col in subs]
    rings = tuple([rot(p) for p in ring] for ring in rings)

    # Fit the whole glyph into the padded box.
    pts = [p for b in bonds for p in b] + [s[0] for s in subs]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    span = max(hi_x - lo_x, hi_y - lo_y)
    target = size * (1 - 2 * PAD)
    k = target / span
    ox = size / 2 - (lo_x + hi_x) / 2 * k
    oy = size / 2 - (lo_y + hi_y) / 2 * k

    def T(p):
        return (p[0] * k + ox, p[1] * k + oy)

    bond_w = max(2, int(size * 0.016 * k))
    node_r = size * 0.034 * k

    for a, b in bonds:
        d.line([T(a), T(b)], fill=BOND, width=bond_w)

    # Aromatic accents: inset parallel lines on alternating edges.
    for ring in rings:
        mx = sum(p[0] for p in ring) / 6
        my = sum(p[1] for p in ring) / 6
        for i in (0, 2, 4):
            a, b = ring[i], ring[(i + 1) % 6]
            a2 = (a[0] + (mx - a[0]) * 0.22 + (b[0] - a[0]) * 0.18,
                  a[1] + (my - a[1]) * 0.22 + (b[1] - a[1]) * 0.18)
            b2 = (b[0] + (mx - b[0]) * 0.22 - (b[0] - a[0]) * 0.18,
                  b[1] + (my - b[1]) * 0.22 - (b[1] - a[1]) * 0.18)
            d.line([T(a2), T(b2)], fill=BOND, width=int(bond_w * 0.6))

    # Carbons, de-duplicated at the shared (fusion) vertices.
    seen: set[tuple[int, int]] = set()
    for p in nodes:
        q = T(p)
        key = (round(q[0] / 10), round(q[1] / 10))
        if key in seen:
            continue
        seen.add(key)
        disc(d, q, node_r, NODE_C)

    for p, col in subs:
        disc(d, T(p), node_r * 1.15, col)

    img.resize((S, S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({S}x{S}, {os.path.getsize(OUT)/1024:.0f} KB)")


def disc(d: ImageDraw.ImageDraw, p, r: float, col) -> None:
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=col)


if __name__ == "__main__":
    main()
