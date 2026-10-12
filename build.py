"""Build index.html from src/template.html + src/meta.json + data/routes.json."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(ROOT, 'src', 'template.html'), encoding='utf-8').read()
meta = json.load(open(os.path.join(ROOT, 'src', 'meta.json'), encoding='utf-8'))
geo = json.load(open(os.path.join(ROOT, 'data', 'routes.json'), encoding='utf-8'))

missing = [r['id'] for r in meta['routes'] if r['id'] not in geo]
assert not missing, f'no geometry for {missing}; run tools/route.py'
geo = {r['id']: geo[r['id']] for r in meta['routes']}

dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
html = tpl.replace('__META__', dump(meta)).replace('__ROUTES__', dump(geo))
open(os.path.join(ROOT, 'index.html'), 'w', encoding='utf-8').write(html)
print(f'index.html {len(html) // 1024} KB, {len(meta["routes"])} routes')
