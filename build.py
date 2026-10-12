"""Build index.html from src/template.html + src/meta.json + data/routes.json."""
import json, math, os

ROOT = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(ROOT, 'src', 'template.html'), encoding='utf-8').read()
meta = json.load(open(os.path.join(ROOT, 'src', 'meta.json'), encoding='utf-8'))
geo = json.load(open(os.path.join(ROOT, 'data', 'routes.json'), encoding='utf-8'))

missing = [r['id'] for r in meta['routes'] if r['id'] not in geo]
assert not missing, f'no geometry for {missing}; run tools/route.py'
geo = {r['id']: geo[r['id']] for r in meta['routes']}


def seg_km(pts):
    k = math.cos(math.radians(63.3))
    return sum(math.hypot((b[0] - a[0]) * 111.13, (b[1] - a[1]) * 111.32 * k) for a, b in zip(pts, pts[1:]))


def nearest(line, p):
    return min(range(len(line)), key=lambda i: (line[i][0] - p[0]) ** 2 + ((line[i][1] - p[1]) * 0.45) ** 2)


# Cut each from/to spot out of its route line; add the router's walking stretches as spots.
for r in meta['routes']:
    line = geo[r['id']]['line']
    for sp in r['spots']:
        if 'from' in sp:
            i, j = sorted((nearest(line, sp['from']), nearest(line, sp['to'])))
            sp['line'] = line[i:j + 1]
            sp['km'] = round(seg_km(sp['line']), 1)
    for w in geo[r['id']]['walks']:
        km = seg_km(w)
        if km >= 0.5:
            r['spots'].append({'type': 'walk', 'name': 'On foot', 'line': w, 'km': round(km, 1),
                               'text': 'No ice mapped here, so this is walking (to or from the ice, or between lakes).'})
    r['km_walk'] = round(sum(sp.get('km', 0) for sp in r['spots'] if sp['type'] == 'walk'), 1)
    r['km_maybe'] = round(sum(sp.get('km', 0) for sp in r['spots'] if sp['type'] in ('maybe', 'slow')), 1)

dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
html = tpl.replace('__META__', dump(meta)).replace('__ROUTES__', dump(geo))
open(os.path.join(ROOT, 'index.html'), 'w', encoding='utf-8').write(html)
print(f'index.html {len(html) // 1024} KB, {len(meta["routes"])} routes')
