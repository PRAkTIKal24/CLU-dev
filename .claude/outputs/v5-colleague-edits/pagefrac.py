import subprocess, re, sys
pdf = sys.argv[1]
out = subprocess.run(["pdftotext","-bbox",pdf,"-"],capture_output=True,text=True).stdout
pages = out.split("<page ")[1:]
def words(p):
    return [(float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4)),m.group(5))
            for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>',p)]
# reference full page: page 4 (index 3)
ref = words(pages[3])
top = min(w[1] for w in ref)
bot = max(w[3] for w in ref if w[4] != '4')   # exclude folio
# find last main-text word on page 5: the one before "References"
p5 = words(pages[4])
idx = [i for i,w in enumerate(p5) if w[4]=="References"]
i = idx[0]
last = p5[i-1]
frac = (last[3]-top)/(bot-top)
print(f"page4 body top={top:.2f} bot={bot:.2f} height={bot-top:.2f}")
print(f"last main-text word on p5: {last[4]!r} yMax={last[3]:.2f}")
print(f"MAIN TEXT PAGES = 4 + {frac:.4f} = {4+frac:.2f} pp")
