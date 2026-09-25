"""The PW-2 item-2 layout. Entries: an int = a v2 paragraph chunk id (blocks.py numbering);
('S', k) = a style-only chunk (style numbering); a str = new heading text (the only new text).
The style twin's layout is derived: v2 id i -> style id  i (i<=8), i+1 (9..54), i+2 (55..61)."""
RULE = '% ' + '=' * 87
def sec(title, label):
    return f'{RULE}\n\\section{{{title}}}\n{label}\n{RULE}'
def sub(title, label=None):
    return f'\\subsection{{{title}}}' + (f'\n{label}' if label else '')
SLOT = '% PW2-ITEM3-SLOT'
LAYOUT = [
    sec('Introduction', r'\label{sec:intro}'),
    sub('Positioning among naive units', r'\label{sec:positioning}'), 0, 1, 2, 3, 4,
    sub('The three properties', r'\label{sec:why}'), 5,
    sub('Contributions', r'\label{sec:contrib}'), 6, 7,
    sec('Methods', r'\label{sec:object}'),
    SLOT, ('S', 9),
    9, 10, 14, 50, 16, 11, 12, 13,
    sec('Experiments', r'\label{sec:experiments}'),
    8,
    sub('The Shapes3D value hold-out', r'\label{sec:shapes}'), 17,
    sub('Baselines and the matching rule'), 15, 24,
    sub('The $n$-way ladder protocol'), 26,
    sub('The deletion protocol'), 32,
    sub('The CMAZE-T transition task', r'\label{sec:maze}'), 47, 48,
    sec('Results', r'\label{sec:results}'),
    sub('Composition on Shapes3D', r'\label{sec:shapes-t7}'), 18, 19, 20, 21, 22, 23, 25,
    sub('The $n$-way ladder', r'\label{sec:ladder}'), 27, 28, 29, 30, 31,
    sub('Deletion on Shapes3D', r'\label{sec:shapes-del}'), 33, 34, 35, 36,
    sub('Deletion against retrained baselines', r'\label{sec:del-comparators}'), 37,
    sub('Refusal through the hole', r'\label{sec:hole}'), 38, 39, 40,
    sub('Two ablations', r'\label{sec:ablations}'), 41, 42, 43,
    sub('Encoder adaptation and generative baselines', r'\label{sec:adapt}\label{sec:bard}'), 44, 45, 46,
    sub('CMAZE-T conditions and relational coding', r'\label{sec:maze-wm}'), 49, 51, 52, 53,
    sec('Discussion', r'\label{sec:limits}'),
    ('S', 56),
    sub('Limitations', r'\label{sec:limits-list}'), 55,
    sub('A planned CMAZE suite'), 54,
    sub('The zero-dynamics clue', r'\label{sec:clue}'), 56,
    sub('The Latent Dynamics Model class', r'\label{sec:theory}'), 57, 58,
    sub('Summary', r'\label{sec:leaving}'), 59,
    sub('Physics follow-up', r'\label{sec:followup}'), 60, 61,
]
# where each v2 chunk came from (old section / subsection), for MOVE-MAP
def style_id(i):
    return i if i <= 8 else (i + 1 if i <= 54 else i + 2)
def layout_for(which):
    out = []
    for x in LAYOUT:
        if isinstance(x, int): out.append(('P', x if which == 'v2' else style_id(x)))
        elif isinstance(x, tuple):
            if which == 'style': out.append(('P', x[1]))
        else: out.append(('T', x))
    return out
