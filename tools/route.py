"""Route skating trips over OSM water (lakes + stream buffers) with walking portages.

Reads research/osm/{water_*,streams*}.json and research/strava/tracks.json,
writes data/routes.json: per route a simplified line plus ice/walk km.
"""
import glob, json, math, os
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import polygonize, unary_union
from skimage.graph import MCP_Geometric
from skimage.draw import polygon as draw_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'research')
lakes = {(e['type'], e['id']): e for f in sorted(glob.glob(f'{R}/osm/water_*.json')) for e in json.load(open(f))['elements']}.values()
streams = [e for f in sorted(glob.glob(f'{R}/osm/streams*.json')) for e in json.load(open(f))['elements']]
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

def tor(out):
    """Out and back: the same line there and home again."""
    return route(out + out[-2::-1])

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
    'rensjon-tor': lambda: tor([(63.70768, 12.35584), (63.70631, 12.36921), (63.70403, 12.37832), (63.69998, 12.38679), (63.69350, 12.40400), (63.68750, 12.41350), (63.68358, 12.43716), (63.67400, 12.43860), (63.67078, 12.44366), (63.66740, 12.46010), (63.66393, 12.47469), (63.66194, 12.50393), (63.65550, 12.52880), (63.64700, 12.52300)]),
    'rektjarnarna-tor': lambda: tor([(63.25310, 12.18340), (63.25060, 12.18865), (63.24730, 12.18652), (63.24300, 12.18019), (63.23817, 12.17500), (63.23264, 12.15913), (63.22959, 12.15289), (63.22782, 12.15186), (63.22772, 12.14597), (63.22866, 12.13965), (63.22816, 12.13691), (63.22725, 12.14081), (63.22562, 12.14312), (63.22503, 12.13559), (63.22034, 12.12891), (63.22103, 12.12332), (63.21921, 12.12574), (63.21828, 12.12491), (63.21829, 12.12235), (63.22047, 12.12060), (63.21881, 12.11771), (63.22096, 12.11570), (63.22308, 12.11087), (63.22331, 12.10641), (63.22465, 12.10697), (63.22535, 12.10401), (63.22960, 12.10143), (63.22823, 12.09691), (63.22962, 12.09501), (63.22869, 12.08623), (63.22923, 12.08377), (63.23052, 12.08198), (63.23003, 12.07907), (63.23042, 12.07756), (63.23170, 12.07618), (63.23194, 12.07852), (63.23248, 12.07651), (63.23321, 12.07753), (63.23383, 12.07484), (63.23544, 12.07645), (63.23579, 12.07557), (63.23956, 12.07446), (63.24034, 12.07381), (63.24043, 12.06746)]),
    'norder-rensjon-tor': lambda: tor([(63.31582, 12.23063), (63.31219, 12.25424), (63.31666, 12.26877), (63.31686, 12.28897), (63.32400, 12.29650), (63.33046, 12.29600), (63.34088, 12.29230), (63.35268, 12.28607), (63.35452, 12.28830), (63.35704, 12.28846), (63.36165, 12.28523), (63.36835, 12.28461), (63.37006, 12.28363), (63.37414, 12.28084), (63.37869, 12.27465), (63.38541, 12.26921), (63.39168, 12.25900), (63.39792, 12.24719), (63.40421, 12.23777), (63.40904, 12.22769), (63.41320, 12.22007)]),
    'ottsjo-hackren': lambda: route([(63.2094, 13.0563), (63.20786, 13.05700), (63.20192, 13.07020), (63.20084, 13.08219), (63.20084, 13.10178), (63.20336, 13.12857), (63.20606, 13.16136), (63.20120, 13.17216), (63.19886, 13.19975), (63.20048, 13.23414), (63.20300, 13.25413), (63.21128, 13.29012), (63.21487, 13.31531), (63.21487, 13.38128), (63.20588, 13.41967), (63.19130, 13.45605), (63.18500, 13.51963), (63.18500, 13.57761), (63.18860, 13.61040), (63.19238, 13.61880)]),
    'anjan-runt': lambda: route([(63.70872, 12.62973), (63.69900, 12.60974), (63.69990, 12.57975), (63.70998, 12.55816), (63.71340, 12.54216), (63.71340, 12.48299), (63.71268, 12.47019), (63.71808, 12.48978), (63.73013, 12.48938), (63.73787, 12.47299), (63.74507, 12.45260), (63.75191, 12.44500), (63.74003, 12.46579), (63.72600, 12.50138), (63.72600, 12.52977), (63.72618, 12.55816), (63.71538, 12.56975), (63.70494, 12.58535), (63.70116, 12.61374), (63.70800, 12.62973)]),
    'lyckan-grapplingen': lambda: tor([(63.31508, 13.05978), (63.28744, 12.98688), (63.28668, 12.98675), (63.28434, 12.97395), (63.28236, 12.96995), (63.28020, 12.96635), (63.27354, 12.95276), (63.27444, 12.93597), (63.27786, 12.92837), (63.27966, 12.90838), (63.28460, 12.89870), (63.29154, 12.88718), (63.29244, 12.86959), (63.29532, 12.85999), (63.29568, 12.85560), (63.29892, 12.84880), (63.29532, 12.84200), (63.29244, 12.84240)]),
    'naldsjon-ange': lambda: tor([(63.34870, 14.25417), (63.35368, 14.25900), (63.35450, 14.26300), (63.35600, 14.27000), (63.36400, 14.27200), (63.37500, 14.26000), (63.38700, 14.23000), (63.39300, 14.20000), (63.39700, 14.17000), (63.40200, 14.14500), (63.40750, 14.12500), (63.41250, 14.11400), (63.42000, 14.10000), (63.42750, 14.09300), (63.43500, 14.07900), (63.44250, 14.07200), (63.45000, 14.06800), (63.45317, 14.06791)]),
    'trangsviken-ostersund': lambda: route([(63.33382, 14.00317), (63.33150, 14.00600), (63.32900, 14.01100), (63.32650, 14.02000), (63.32450, 14.03000), (63.31000, 14.05600), (63.29500, 14.07800), (63.26500, 14.14000), (63.23000, 14.22000), (63.20000, 14.30000), (63.18000, 14.37000), (63.17800, 14.42000), (63.16700, 14.45500), (63.16200, 14.50000), (63.16000, 14.55000), (63.16000, 14.59000), (63.16200, 14.61500), (63.17300, 14.62700), (63.17626, 14.63030)]),
    'trangsviken-svenstavik': lambda: route([(63.33382, 14.00317), (63.33150, 14.00600), (63.32900, 14.01100), (63.32650, 14.02000), (63.32450, 14.03000), (63.31000, 14.05600), (63.29500, 14.07800), (63.26500, 14.14000), (63.23000, 14.20000), (63.19000, 14.24500), (63.16500, 14.27500), (63.14700, 14.28600), (63.13700, 14.29200), (63.13400, 14.29600), (63.12150, 14.32600), (63.11500, 14.34500), (63.10000, 14.38000), (63.07000, 14.40500), (63.04000, 14.43200), (63.00500, 14.46500), (62.96500, 14.47800), (62.92000, 14.48200), (62.87500, 14.47500), (62.84000, 14.47800), (62.81500, 14.47500), (62.80300, 14.45000), (62.79000, 14.44600), (62.78150, 14.43900), (62.78200, 14.43508)]),
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
