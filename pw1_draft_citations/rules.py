"""The citation-pass rules (shared by cite_edit.py and cite_table.py)."""
RULES = [  # (id, kind, pattern, replacement)
    # --- hand-typed author-year -> natbib (Sec. 1.1 positioning) ---
    ('H1', 'convert', r'\(Tenenbaum\s+and\s+Freeman,\s+2000\\nocite\{tenenbaum2000separating\};', r'(\\citealp{tenenbaum2000separating};'),
    ('H2', 'convert', r'TensorFaces,\s+Vasilescu\s+and\s+Terzopoulos,\s+2002\\nocite\{vasilescu2002tensorfaces\}', r'TensorFaces~\\citep{vasilescu2002tensorfaces}'),
    ('H3', 'convert', r'Vlasic\s+et\s+al\.,\s+2005\\nocite\{vlasic2005face\}', r'\\citealp{vlasic2005face}'),
    ('H4', 'convert', r'MLC-VAE,\s+2020~\\citep\{georgopoulos2020multilinear\}', r'MLC-VAE~\\citep{georgopoulos2020multilinear}'),
    ('H5', 'convert', r'GANSpace,\s+2020~\\citep\{harkonen2020ganspace\}', r'GANSpace~\\citep{harkonen2020ganspace}'),
    ('H6', 'convert', r'TCA,\s+2021~\\citep\{oldfield2021tensor\}', r'TCA~\\citep{oldfield2021tensor}'),
    ('H7', 'convert', r'PandA,\s+2023~\\citep\{oldfield2023panda\}', r'PandA~\\citep{oldfield2023panda}'),
    ('H8', 'convert', r'form\s+of\s+Cao\s+and\s+Yang\s+\(2015\)\\nocite\{cao2015towards\}', r'form of~\\citet{cao2015towards}'),
    ('H9', 'convert', r'\$k\$-means\s+\(Ginart\s+et\s+al\.,\s+2019\\nocite\{ginart2019making\}\)', r'$k$-means~\\citep{ginart2019making}'),
    ('H10', 'convert', r'models\s+\(Golatkar,\s+Achille,\s+Swaminathan\s+and\s+Soatto\\nocite\{golatkar2023training\}\)', r'models~\\citep{golatkar2023training}'),
    ('H11', 'convert', r'networks\s+\(Raissi,\s+Perdikaris\s+and\s+Karniadakis,\s+2019\\nocite\{raissi2019physics\}\)', r'networks~\\citep{raissi2019physics}'),
    ('H12', 'convert', r'networks\s+\(Greydanus,\s+Dzamba\s+and\s+Yosinski,\s+2019\\nocite\{greydanus2019hamiltonian\}\)', r'networks~\\citep{greydanus2019hamiltonian}'),
    ('H13', 'convert', r'\(Hall,\s+Marshall\s+and\s+Martin,\s+2002\\nocite\{hall2002adding\},', r'(\\citealp{hall2002adding},'),
    ('H14', 'convert', r'the\s+Blelloch--Golovin\s+stable-matching\s+table~\\citep\{blelloch2007strongly\}', r'the~\\citet{blelloch2007strongly} stable-matching table'),
    # --- citations added at first mention (main text) ---
    ('M1', 'add', r'an\s+LSTM\s+or\s+a\s+transformer\s+is\s+a\s+unit', r'an LSTM~\\citep{hochreiter1997long} or a transformer~\\citep{vaswani2017attention} is a unit'),
    ('M2', 'add', r'LACE\s+and\s+PoE\s+are\s+in', r'LACE~\\citep{nie2021controllable} and PoE~\\citep{hinton2002training} are in'),
    # --- citations added at first mention (appendices) ---
    ('P1', 'add', r'\(LSTM\),', r'(LSTM~\\citep{hochreiter1997long}),'),
    ('P2', 'add', r'\(transformer\)\s+and\s+131,891', r'(transformer~\\citep{vaswani2017attention}) and 131,891'),
    ('P3', 'add', r'\(against\s+LACE,\s+PoE\)', r'(against LACE~\\citep{nie2021controllable}, PoE~\\citep{hinton2002training})'),
    ('P4', 'add', r'at\s+their\s+own\s+settings,\s+Shapes3D\s+value', r'at their own settings, Shapes3D~\\citep{burgess2018shapes,kim2018disentangling} value'),
    ('P5', 'add', r'\(bootstrap\s+2\.5\\%\s+quantile', r'(bootstrap~\\citep{efron1979bootstrap} 2.5\\% quantile'),
    ('P6', 'add', r'Approximate\s+unlearning\s+\(gradient\s+ascent', r'Approximate unlearning~\\citep{guo2020certified,golatkar2020eternal} (gradient ascent'),
]
FIRST = {'M2'}  # first mention in the main text (caption of Tab. t7, before the ladder caption)
BIB = [('B1', r'\IfFileExists{references.bib}{%', r'\IfFileExists{refs_clu_iclr.bib}{%'),
       ('B2', '\\bibliography{references}\n', '\\bibliography{refs_clu_iclr}\n'),
       # url breaks at hyphens: the Head's bib has a 222-character URL (lee_fast_2021) that is otherwise 133 pt overfull
       ('B3', '\n\\usepackage{hyperref}\n', '\n\\PassOptionsToPackage{hyphens}{url}\\usepackage{hyperref}\n')]
