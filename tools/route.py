"""Route skating trips over OSM water (lakes + stream buffers) with walking portages.

Reads research/osm/{lakes_geom,streams}.json and research/strava/tracks.json,
writes data/routes.json: per route a simplified line plus ice/walk km.
"""
import json, math, os
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import polygonize, unary_union
from skimage.graph import MCP_Geometric
from skimage.draw import polygon as draw_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'research')
lakes = json.load(open(f'{R}/osm/water_all.json'))['elements']
streams = json.load(open(f'{R}/osm/streams.json'))['elements']
tracks = json.load(open(f'{R}/strava/tracks.json'))

LAT0 = 63.3
KX = 111.32 * math.cos(math.radians(LAT0))  # km per deg lon
KY = 111.13                                 # km per deg lat

def xy(lat, lon): return (lon * KX, lat * KY)
def ll(x, y): return (y / KY, x / KX)

def lake_polys():
    out = []
    for e in lakes:
      try:
          if e['type'] == 'way':
              g = [xy(p['lat'], p['lon']) for p in e['geometry']]
              if len(g) >= 4: out.append(Polygon(g).buffer(0))
          else:
              outer = [LineString([xy(p['lat'], p['lon']) for p in m['geometry']]) for m in e['members'] if m.get('role') == 'outer' and m.get('geometry')]
              inner = [LineString([xy(p['lat'], p['lon']) for p in m['geometry']]) for m in e['members'] if m.get('role') == 'inner' and m.get('geometry')]
              o = unary_union(list(polygonize(unary_union(outer))))
              if inner:
                  o = o.difference(unary_union(list(polygonize(unary_union(inner)))))
              out.append(o)
      except Exception:
        pass
    return unary_union([o for o in out if o.is_valid])

WATER = lake_polys()
STREAMS = unary_union([LineString([xy(p['lat'], p['lon']) for p in e['geometry']]).buffer(0.03) for e in streams])
ICE = unary_union([WATER, STREAMS])

CELL = 0.025  # km
WALK_COST = 6.0

def route(waypoints, pad=1.0):
    pts = [xy(*w) for w in waypoints]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    nx, ny = int((x1 - x0) / CELL) + 1, int((y1 - y0) / CELL) + 1
    ice = np.zeros((ny, nx), bool)
    from shapely.geometry import box
    clip = ICE.intersection(box(x0, y0, x1, y1))
    geoms = getattr(clip, 'geoms', [clip])
    for g in geoms:
        if g.is_empty or g.geom_type != 'Polygon': continue
        ext = np.array(g.exterior.coords)
        rr, cc = draw_poly((ext[:, 1] - y0) / CELL, (ext[:, 0] - x0) / CELL, ice.shape)
        ice[rr, cc] = True
        for hole in g.interiors:
            h = np.array(hole.coords)
            rr, cc = draw_poly((h[:, 1] - y0) / CELL, (h[:, 0] - x0) / CELL, ice.shape)
            ice[rr, cc] = False
    cost = np.where(ice, 1.0, WALK_COST)
    cell = lambda p: (min(ny - 1, max(0, int((p[1] - y0) / CELL))), min(nx - 1, max(0, int((p[0] - x0) / CELL))))
    path = []
    for a, b in zip(pts, pts[1:]):
        m = MCP_Geometric(cost)
        m.find_costs([cell(a)], [cell(b)])
        seg = m.traceback(cell(b))
        path += seg if not path else seg[1:]
    coords = [(x0 + (c + .5) * CELL, y0 + (r + .5) * CELL) for r, c in path]
    flags = [bool(ice[r, c]) for r, c in path]
    ice_km = walk_km = 0.0
    for (p, q), f in zip(zip(coords, coords[1:]), flags[1:]):
        d = math.dist(p, q)
        if f: ice_km += d
        else: walk_km += d
    line = LineString(coords).simplify(0.02)
    out = [list(map(lambda v: round(v, 5), ll(*c))) for c in line.coords]
    # walk segments, for dashed drawing
    walks, cur = [], []
    for c, f in zip(coords, flags):
        if not f: cur.append(c)
        elif cur:
            if len(cur) > 3: walks.append(cur)
            cur = []
    if len(cur) > 3: walks.append(cur)
    walks = [[list(map(lambda v: round(v, 5), ll(*c))) for c in LineString(w).simplify(0.02).coords] for w in walks]
    return dict(line=out, walks=walks, ice_km=round(ice_km, 1), walk_km=round(walk_km, 1))

def track(aid):
    t = tracks[aid]
    p = [xy(*a) for a in t]
    km = sum(math.dist(a, b) for a, b in zip(p, p[1:]))
    return dict(line=[[round(a, 5), round(b, 5)] for a, b in t], walks=[], ice_km=round(km, 1), walk_km=0.0)

ROUTES = {
    'enafors-duved': lambda: track('8203086392'),
    'ann-loop': lambda: track('8144847762'),
    'ann-short': lambda: track('4402779984'),
    'enafors-ann': lambda: route([(63.2898, 12.3375), (63.2836, 12.3305), (63.2724, 12.4776), (63.3213, 12.5391)]),
    'handol-ann': lambda: route([(63.2522, 12.4499), (63.3213, 12.5391)]),
    'ann-ottsjo': lambda: route([(63.3213, 12.5391), (63.2910, 12.6547), (63.2996, 12.7319), (63.2920, 12.8430), (63.2990, 12.8500), (63.2930, 12.8750), (63.2850, 12.9000), (63.2740, 12.9104), (63.2311, 12.9261), (63.2170, 13.0450)]),
    'vala-ottsjo': lambda: route([(63.1487, 12.9653), (63.1572, 12.8992), (63.1756, 12.8455), (63.2717, 12.6441), (63.2910, 12.6547), (63.2996, 12.7319), (63.2920, 12.8430), (63.2990, 12.8500), (63.2930, 12.8750), (63.2850, 12.9000), (63.2740, 12.9104), (63.2311, 12.9261), (63.2170, 13.0450)]),
    'ottsjon': lambda: route([(63.2102, 13.1571), (63.2311, 12.9261), (63.2170, 13.0450)]),
    'liten': lambda: route([(63.3424, 13.4720), (63.3085, 13.6577)]),
    'ann-duved': lambda: route([(63.3213, 12.5391), (63.2910, 12.6547), (63.2996, 12.7319), (63.3137, 12.7871), (63.3207, 12.8250), (63.3327, 12.8229), (63.3528, 12.8682), (63.3667, 12.8783), (63.3900, 12.9176)]),
    'juvuln': lambda: route([(63.7013, 12.9590), (63.7295, 13.1988), (63.6752, 13.3582)]),
    'anjan': lambda: route([(63.7388, 12.4714), (63.7294, 12.5333), (63.7051, 12.6496), (63.6818, 12.8090)]),
    'aresjon': lambda: route([(63.3987, 13.0765), (63.3614, 13.1762)]),
    'gevsjon': lambda: route([(63.3708, 12.6241), (63.3600, 12.6842), (63.3723, 12.6970), (63.3850, 12.6600), (63.3708, 12.6241)]),
    'kallsjon-length': lambda: route([(63.4939, 13.1243), (63.5566, 13.0249), (63.5921, 13.0789), (63.6418, 12.8850), (63.7021, 12.9469)]),
    'klocka-loop': lambda: route([(63.2968, 12.4908), (63.2724, 12.4776), (63.2836, 12.3305), (63.2724, 12.4776), (63.2500, 12.5600), (63.2350, 12.5800), (63.2968, 12.4908)]),
    'helgesjon': lambda: route([(63.3620, 13.2645), (63.3720, 13.2280), (63.3400, 13.3020), (63.3620, 13.2645)]),
    'medstugusjon': lambda: route([(63.5549, 12.3710), (63.5350, 12.3847), (63.5567, 12.3470), (63.5549, 12.3710)]),
    'kallsjon': lambda: route([(63.4885, 13.1224), (63.4749, 13.2364), (63.4127, 13.3701)]),
}

if __name__ == '__main__':
    import sys
    want = sys.argv[1:] or list(ROUTES)
    path = os.path.join(ROOT, 'data', 'routes.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = json.load(open(path)) if os.path.exists(path) else {}
    for k in want:
        out[k] = ROUTES[k]()
        print(k, out[k]['ice_km'], 'ice km,', out[k]['walk_km'], 'walk km,', len(out[k]['line']), 'pts', flush=True)
    json.dump(out, open(path, 'w'), ensure_ascii=False)
