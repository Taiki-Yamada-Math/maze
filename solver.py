"""
手描き迷路ソルバー ― Python コア

紙に手描きした迷路のスキャン画像をグラフに変換し、探索アルゴリズムで解く。
ブラウザ上では Pyodide で、手元では通常の Python（numpy と Pillow）で動く。

処理の流れ
  1. prepare()  画像 → 壁の格子 → グラフ（通路の各セルがノード）
  2. solve()    スタートからゴールまでを BFS / DFS / ダイクストラ / A* で探索

コマンドラインからも使える:
  python solver.py maze.png --start 40,30 --goal 900,650 --algo astar -o solved.png
"""

import heapq
import json
import math
import time
from collections import deque

import numpy as np

SQ2 = math.sqrt(2.0)
CLEARANCE = 1.5   # 壁のそばを通る辺ほど重くする係数（ダイクストラ / A* で使う）
MAX_DIST = 40     # 壁からの距離はこれ以上を区別しない

_STATE = {}


# ---------------------------------------------------------------- 画像処理

def _box_mean(a, r):
    """半径 r の平均フィルタ（積分画像で高速化）。"""
    p = np.pad(a, r, mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _otsu(values):
    """大津の方法で二値化のしきい値を求める。"""
    hist, edges = np.histogram(values, bins=128)
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    denom = w * (1 - w)
    denom[denom == 0] = np.inf
    between = (mu[-1] * w - mu) ** 2 / denom
    return float(edges[int(np.argmax(between)) + 1])


def _dilate(mask, times):
    """3×3 の膨張を times 回。途切れた線をつなぐのに使う。"""
    m = mask
    for _ in range(times):
        p = np.pad(m, 1)
        m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:] |
             p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:] |
             p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return m


def _components(mask):
    """8 近傍でつながった壁のかたまりを列挙する。"""
    gh, gw = mask.shape
    flat = mask.ravel().tolist()
    seen = bytearray(gh * gw)
    comps = []
    for i in range(gh * gw):
        if not flat[i] or seen[i]:
            continue
        seen[i] = 1
        stack, cells = [i], []
        r0 = r1 = i // gw
        c0 = c1 = i % gw
        while stack:
            p = stack.pop()
            cells.append(p)
            r, c = divmod(p, gw)
            if r < r0: r0 = r
            if r > r1: r1 = r
            if c < c0: c0 = c
            if c > c1: c1 = c
            for dr in (-1, 0, 1):
                rr = r + dr
                if rr < 0 or rr >= gh:
                    continue
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if cc < 0 or cc >= gw:
                        continue
                    q = rr * gw + cc
                    if flat[q] and not seen[q]:
                        seen[q] = 1
                        stack.append(q)
        comps.append({"cells": cells, "size": len(cells), "box": (r0, c0, r1, c1)})
    return comps


def _distance_to_wall(walls):
    """各セルから最寄りの壁までの距離（チェビシェフ距離）。"""
    d = np.full(walls.shape, MAX_DIST, dtype=np.int32)
    d[walls] = 0
    reached = walls.copy()
    for k in range(1, MAX_DIST):
        grown = _dilate(reached, 1)
        new = grown & ~reached
        if not new.any():
            break
        d[new] = k
        reached |= new
    return d


def _to_grid(pen, h, w, grid_max):
    """インクの画素を格子にまとめ、ゴミと用紙のふちの影を除き、迷路の範囲を決める。"""
    f = max(1, math.ceil(max(h, w) / int(grid_max)))
    gh, gw = math.ceil(h / f), math.ceil(w / f)
    padded = np.zeros((gh * f, gw * f), dtype=bool)
    padded[:h, :w] = pen
    # 細い線が消えないよう、セルの 12% 以上がインクなら壁とする
    walls = padded.reshape(gh, f, gw, f).mean(axis=(1, 3)) >= 0.12

    kept = []
    for comp in _components(walls):
        r0, c0, r1, c1 = comp["box"]
        touches_edge = r0 == 0 or c0 == 0 or r1 == gh - 1 or c1 == gw - 1
        thin = min((r1 - r0 + 1) / gh, (c1 - c0 + 1) / gw) < 0.04
        if comp["size"] < 4 or (touches_edge and thin):
            for p in comp["cells"]:
                walls.flat[p] = False
        else:
            kept.append(comp)
    if not kept:
        return None

    # 主要な線を囲む四角形の外側は壁にして、迷路の外回りを禁止する
    largest = max(c["size"] for c in kept)
    main = [c for c in kept if c["size"] >= 0.02 * largest]
    r0 = min(c["box"][0] for c in main); c0 = min(c["box"][1] for c in main)
    r1 = max(c["box"][2] for c in main); c1 = max(c["box"][3] for c in main)
    outside = np.ones_like(walls)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    return walls, outside, (r0, c0, r1, c1), f, gh, gw


def _corridor_half_width(walls):
    """通路の半幅（セル単位）を推定する。通路の中心線上の、壁までの距離の中央値。"""
    d = _distance_to_wall(np.pad(walls, 1, constant_values=True))[1:-1, 1:-1]
    p = np.pad(d, 1)
    ridge = (d > 0)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == 1 and dc == 1:
                continue
            ridge &= d >= p[dr:dr + d.shape[0], dc:dc + d.shape[1]]
    vals = d[ridge]
    return float(np.median(vals)) if vals.size else 3.0


def prepare(gray_buf, h, w, sensitivity=5, grow=-1, grid_max=0):
    """
    グレースケール画像（h×w, uint8）から迷路のグラフを作る。

    sensitivity : 1〜9。大きいほど薄い線まで壁とみなす
    grow        : 壁を太らせる回数。ペンのかすれによる隙間をふさぐ。-1 で自動
    grid_max    : 格子の長辺のセル数。0 で自動（細かい迷路ほど細かくする）
    """
    t0 = time.perf_counter()
    h, w = int(h), int(w)
    data = gray_buf.to_bytes() if hasattr(gray_buf, "to_bytes") else bytes(gray_buf)
    gray = np.frombuffer(data, dtype=np.uint8).reshape(h, w).astype(np.float64)

    # 1. 周りの紙より暗い部分をインクとみなす（影や照明ムラに強い）
    radius = max(8, round(max(h, w) / 30))
    ink = np.clip(_box_mean(gray, radius) - gray, 0, None)
    scale = 1.5 - 0.125 * (int(sensitivity) - 1)
    threshold = max(_otsu(ink) * scale, 12.0)
    pen = ink > threshold

    # 2〜4. 格子化・ゴミ除去・迷路範囲の決定。細かい迷路なら解像度を上げてやり直す
    gm = int(grid_max) if int(grid_max) > 0 else 260
    built = _to_grid(pen, h, w, gm)
    if built is None:
        return json.dumps({"ok": False, "error": "迷路の線が見つかりませんでした。感度を上げてください。"})
    walls, outside, box, f, gh, gw = built
    half = _corridor_half_width(walls | outside)
    if int(grid_max) <= 0 and half < 6:
        gm2 = int(min(420, round(gm * 7 / max(half, 1.0))))
        if gm2 > gm:
            rebuilt = _to_grid(pen, h, w, gm2)
            if rebuilt is not None:
                walls, outside, box, f, gh, gw = rebuilt
                half = _corridor_half_width(walls | outside)
    r0, c0, r1, c1 = box

    # 5. ペンのかすれで生じた隙間をふさぐ。通路の幅に比べて十分小さい隙間だけを
    #    閉じたいので、補強量は通路の半幅から自動で決める（手動指定も可）
    auto_grow = int(min(5, max(1, round(0.4 * half))))
    grow = auto_grow if int(grow) < 0 else int(grow)
    shown = _dilate(walls, grow)          # 画面に重ねる壁（範囲外の塗りつぶしは含めない）
    walls = shown | outside

    # 6. グラフ化：通路のセルがノード。外周に 1 セルの壁を足して境界判定を省く
    wp = np.pad(walls, 1, constant_values=True)
    free = ~wp
    dist = _distance_to_wall(wp)
    right = int((free[:, :-1] & free[:, 1:]).sum())
    down = int((free[:-1, :] & free[1:, :]).sum())
    blocks = int((free[:-1, :-1] & free[:-1, 1:] & free[1:, :-1] & free[1:, 1:]).sum())

    _STATE.clear()
    _STATE.update(
        gh=gh, gw=gw, W=gw + 2,
        free=free.ravel().tolist(),
        mult=(1.0 + CLEARANCE / np.maximum(dist, 1)).ravel().tolist(),
    )
    return json.dumps({
        "ok": True, "gh": gh, "gw": gw, "f": f,
        "walls": (shown.ravel().astype(np.uint8) + 48).tobytes().decode("ascii"),
        "box": [r0, c0, r1, c1],
        "nodes": int(free.sum()), "edges": right + down + 2 * blocks,
        "threshold": round(threshold, 1),
        "grow": grow, "auto_grow": auto_grow, "half_width": round(half, 1),
        "ms": round((time.perf_counter() - t0) * 1000),
    })


# ---------------------------------------------------------------- 探索

def _snap(r, c):
    """壁の上がクリックされたら、いちばん近い通路のセルに移す。"""
    gh, gw, W, free = _STATE["gh"], _STATE["gw"], _STATE["W"], _STATE["free"]
    r = min(max(int(r), 0), gh - 1)
    c = min(max(int(c), 0), gw - 1)
    start = (r + 1) * W + (c + 1)
    if free[start]:
        return start
    seen = {start}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for o in (-W, W, -1, 1):
            q = p + o
            if 0 <= q < len(free) and q not in seen:
                if free[q]:
                    return q
                seen.add(q)
                queue.append(q)
    return None


def _neighbors(p, W, free):
    """p から動ける隣のセルと移動距離。斜めは角をすり抜けない場合だけ許す。"""
    out = []
    for o in (-W, W, -1, 1):
        if free[p + o]:
            out.append((p + o, 1.0))
    for o, a, b in ((-W - 1, -W, -1), (-W + 1, -W, 1), (W - 1, W, -1), (W + 1, W, 1)):
        if free[p + o] and free[p + a] and free[p + b]:
            out.append((p + o, SQ2))
    return out


def _bfs(s, g, W, free):
    parent = {s: -1}
    order = []
    queue = deque([s])
    while queue:
        p = queue.popleft()
        order.append(p)
        if p == g:
            return order, parent, True
        for q, _ in _neighbors(p, W, free):
            if q not in parent:
                parent[q] = p
                queue.append(q)
    return order, parent, False


def _dfs(s, g, W, free):
    parent = {}
    order = []
    stack = [(s, -1)]
    while stack:
        p, frm = stack.pop()
        if p in parent:
            continue
        parent[p] = frm
        order.append(p)
        if p == g:
            return order, parent, True
        for q, _ in reversed(_neighbors(p, W, free)):
            if q not in parent:
                stack.append((q, p))
    return order, parent, False


def _best_first(s, g, W, free, mult, use_heuristic):
    """ダイクストラ法（use_heuristic=False）と A*（True）。"""
    gr, gc = divmod(g, W)

    def h(p):
        if not use_heuristic:
            return 0.0
        r, c = divmod(p, W)
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (SQ2 - 2) * min(dr, dc)   # 8 方向の直線距離（重み 1 以上なので許容的）

    cost = {s: 0.0}
    parent = {s: -1}
    closed = set()
    order = []
    heap = [(h(s), 0.0, s)]
    while heap:
        _, d, p = heapq.heappop(heap)
        if p in closed:
            continue
        closed.add(p)
        order.append(p)
        if p == g:
            return order, parent, True, cost
        for q, step in _neighbors(p, W, free):
            if q in closed:
                continue
            nd = d + step * mult[q]
            if nd < cost.get(q, math.inf):
                cost[q] = nd
                parent[q] = p
                heapq.heappush(heap, (nd + h(q), nd, q))
    return order, parent, False, cost


def solve(sr, sc, gr, gc, algo="bfs"):
    """スタート (sr, sc) からゴール (gr, gc) まで探索する。座標は格子のセル単位。"""
    if not _STATE:
        return json.dumps({"ok": False, "error": "先に迷路を読み込んでください。"})
    t0 = time.perf_counter()
    gw, W = _STATE["gw"], _STATE["W"]
    free, mult = _STATE["free"], _STATE["mult"]
    s, g = _snap(sr, sc), _snap(gr, gc)
    if s is None or g is None:
        return json.dumps({"ok": False, "error": "通路が見つかりません。"})

    cost = None
    if algo == "bfs":
        order, parent, found = _bfs(s, g, W, free)
    elif algo == "dfs":
        order, parent, found = _dfs(s, g, W, free)
    elif algo in ("dijkstra", "astar"):
        order, parent, found, cost = _best_first(s, g, W, free, mult, algo == "astar")
    else:
        return json.dumps({"ok": False, "error": f"未知のアルゴリズム: {algo}"})

    path = []
    if found:
        p = g
        while p != -1:
            path.append(p)
            p = parent[p]
        path.reverse()

    def unpad(p):
        r, c = divmod(p, W)
        return (r - 1) * gw + (c - 1)

    length = sum(SQ2 if abs(a - b) not in (1, W) else 1.0 for a, b in zip(path, path[1:]))
    return json.dumps({
        "ok": True, "found": found, "algo": algo,
        "start": divmod(unpad(s), gw), "goal": divmod(unpad(g), gw),
        "order": [unpad(p) for p in order],
        "path": [unpad(p) for p in path],
        "visited": len(order),
        "steps": max(len(path) - 1, 0),
        "length": round(length, 1),
        "cost": round(cost[g], 1) if (found and cost is not None) else None,
        "ms": round((time.perf_counter() - t0) * 1000),
    })


# ---------------------------------------------------------------- コマンドライン

def _main():
    import argparse
    from PIL import Image, ImageDraw

    ap = argparse.ArgumentParser(description="手描き迷路の画像を解く")
    ap.add_argument("image")
    ap.add_argument("--start", required=True, help="スタートの画素座標 x,y")
    ap.add_argument("--goal", required=True, help="ゴールの画素座標 x,y")
    ap.add_argument("--algo", default="astar", choices=["bfs", "dfs", "dijkstra", "astar"])
    ap.add_argument("--sensitivity", type=int, default=5)
    ap.add_argument("--grow", type=int, default=-1, help="-1 で自動")
    ap.add_argument("-o", "--output", default="solved.png")
    a = ap.parse_args()

    img = Image.open(a.image).convert("RGB")
    k = min(1.0, 1000 / max(img.size))
    work = img.resize((round(img.width * k), round(img.height * k)))
    gray = np.asarray(work.convert("L"), dtype=np.uint8)
    info = json.loads(prepare(gray.tobytes(), gray.shape[0], gray.shape[1], a.sensitivity, a.grow))
    if not info["ok"]:
        raise SystemExit(info["error"])
    f = info["f"]
    sx, sy = (float(v) * k for v in a.start.split(","))
    gx, gy = (float(v) * k for v in a.goal.split(","))
    res = json.loads(solve(sy // f, sx // f, gy // f, gx // f, a.algo))
    print(f"グラフ: ノード {info['nodes']:,} / 辺 {info['edges']:,}")
    if not res["found"]:
        raise SystemExit(f"経路が見つかりません（{res['visited']:,} セルを探索）")
    print(f"{a.algo}: {res['visited']:,} セルを探索、{res['steps']} ステップ（{res['ms']} ms）")

    draw = ImageDraw.Draw(img)
    scale = f / k
    pts = [((p % info["gw"] + 0.5) * scale, (p // info["gw"] + 0.5) * scale) for p in res["path"]]
    draw.line(pts, fill=(240, 160, 0), width=max(3, round(scale * 0.8)), joint="curve")
    img.save(a.output)
    print(f"保存しました: {a.output}")


if __name__ == "__main__":
    _main()
