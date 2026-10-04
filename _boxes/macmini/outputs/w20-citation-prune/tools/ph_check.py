import sys,re,pathlib,hashlib
def args(t, cmd):
    res=[];i=0
    while True:
        j=t.find('\\'+cmd+'{',i)
        if j<0: return res
        k=j+len(cmd)+2;d=1
        while d: d+= {'{':1,'}':-1}.get(t[k],0); k+=1
        res.append(t[j:k]); i=k
b,a=map(pathlib.Path,sys.argv[1:3])
for n in ["DRAFT-v2.tex","DRAFT-v2-style.tex"]:
    for c in ["PLACEHOLDER","TODO"]:
        pb,pa=args((b/n).read_text(),c),args((a/n).read_text(),c)
        print(n,c,len(pb),"->",len(pa),"ordered byte-identical:",pb==pa, hashlib.sha256("".join(pa).encode()).hexdigest()[:16])
