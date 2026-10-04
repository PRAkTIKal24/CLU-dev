import sys, pypdf, difflib
def t(p): return pypdf.PdfReader(p).pages[0].extract_text().splitlines()
a=t(sys.argv[1]); b=t(sys.argv[2])
print("# rendered text of fig_ladder.pdf (pypdf), installed 949d5b6d... -> rebuilt 5211de78...")
for l in difflib.unified_diff(a,b,"installed","rebuilt",lineterm="",n=1): print(l)
print("# lines before", len(a), "after", len(b))
