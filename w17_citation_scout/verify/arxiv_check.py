import sys, json, time, urllib.request, xml.etree.ElementTree as ET
ids=[l.split()[0] for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
out={}
for i in range(0,len(ids),40):
    chunk=ids[i:i+40]
    url='https://export.arxiv.org/api/query?max_results=100&id_list='+','.join(chunk)
    for attempt in range(4):
        try:
            data=urllib.request.urlopen(url,timeout=60).read(); break
        except Exception as e:
            time.sleep(5)
    root=ET.fromstring(data)
    for e in root.findall('a:entry',ns):
        aid=e.find('a:id',ns).text.split('/abs/')[-1]
        base=aid.rsplit('v',1)[0]
        doi=e.find('x:doi',ns); jr=e.find('x:journal_ref',ns); cm=e.find('x:comment',ns)
        out[base]={'title':' '.join(e.find('a:title',ns).text.split()),
          'first_author':e.find('a:author/a:name',ns).text,
          'published':e.find('a:published',ns).text[:10],
          'doi':doi.text if doi is not None else None,
          'journal_ref':jr.text if jr is not None else None,
          'comment':' '.join(cm.text.split())[:160] if cm is not None else None}
    time.sleep(3)
for a in ids:
    print(json.dumps({a:out.get(a,'NOT FOUND')},ensure_ascii=False))
