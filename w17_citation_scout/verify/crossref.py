import sys,json,urllib.request,urllib.parse,time,re
H={'User-Agent':'citation-scout/1.0 (mailto:info@forgis.com)'}
def get(u):
    for a in range(3):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=40))['message']
        except Exception as e: err=e; time.sleep(3)
    return {'ERR':str(err)}
def fmt(m):
    if 'ERR' in m: return m['ERR']
    t=(m.get('title') or ['?'])[0]; a=(m.get('author') or [{}])[0].get('family','?')
    y=(m.get('issued',{}).get('date-parts') or [[None]])[0][0]; c=(m.get('container-title') or [''])[0]
    return f"{t[:90]} | {a} | {y} | {c[:50]} | {m.get('DOI')}"
mode=sys.argv[1]
for line in open(sys.argv[2]):
    line=line.strip()
    if not line or line.startswith('#'): continue
    if mode=='doi':
        print(line,'=>',fmt(get('https://api.crossref.org/works/'+urllib.parse.quote(line))))
    else:
        items=get('https://api.crossref.org/works?rows=3&query.bibliographic='+urllib.parse.quote(line)).get('items',[])
        print('Q:',line); [print('   ',fmt(m)) for m in items]
    time.sleep(1)
