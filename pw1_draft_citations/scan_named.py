"""usage: scan_named.py DRAFT.tex -- for each named work/method/dataset/model term: first mention in the
main text (after the abstract) and in the appendices, with whether a \\cite* sits in the same sentence.
Comments are stripped (a % not preceded by a backslash)."""
import re, sys

TERMS = {  # term regex -> bib key(s) that name it ('' = not in the bib)
    r'Shapes3D|3D Shapes|3dshapes': '3dshapes18 kim_disentangling_2019',
    r'\btransformers?\b|\bTransformer\b': 'vaswani_attention_2023',
    r'\bLSTMs?\b': 'hochreiter_long_1997',
    r'Mamba': '', r'DeltaNet|Delta ?Net': '', r'\bRWKV\b': '', r'\bS4\b': '',
    r'\bLACE\b': 'nie_controllable_2021', r'\bPoE\b|[Pp]roduct[- ]of[- ][Ee]xperts': 'hinton_training_2002',
    r'\bSISA\b': 'bourtoule_machine_2021', r'OpenAlex': 'priem_openalex_2022',
    r'[Bb]ootstrap': 'efron_bootstrap_1979', r'Hopfield': 'hopfield_neural_1982 ramsauer_hopfield_2021',
    r'[Dd]ense associative': 'krotov_dense_2016', r'[Ee]nergy [Tt]ransformer': 'hoover_energy_2023',
    r'[Rr]eservoir': 'kong_reservoir-computing_2024', r'GANSpace': 'harkonen_ganspace_2020',
    r'TensorFaces': 'goos_multilinear_2002', r'\bTCA\b': 'oldfield_tensor_2021', r'PandA': 'oldfield_panda_2023',
    r'MLC-VAE': 'georgopoulos_multilinear_2020', r'GMOR|Gell-Mann': 'gell-mann_behavior_1968',
    r'Blelloch': 'blelloch_strongly_2007', r'[Pp]hysics-informed': 'raissi_physics-informed_2019',
    r'Hamiltonian neural': 'greydanus_hamiltonian_2019', r'[Ww]orld model': 'ha_world_2018 hafner_mastering_2025',
    r'Dreamer': 'hafner_mastering_2025', r'GameNGen|[Gg]ame engine': 'valevski_diffusion_2025',
    r'kNN-LM|nearest[- ]neighbou?r language': 'khandelwal_generalization_2020', r'RETRO': 'borgeaud_improving_2022',
    r'[Pp]roduct[- ]key': 'lample_large_2019', r'[Cc]ertified (data )?removal|\(\\varepsilon, ?\\delta\)': 'guo_certified_2023',
    r'[Ss]elective forgetting|Fisher forgetting|Eternal': 'golatkar_eternal_2020',
    r'[Cc]ompartmentali[sz]ed diffusion': 'golatkar_training_2024', r'[Cc]omposable diffusion': 'avidan_compositional_2022',
    r'[Ee]nergy[- ]based model': 'du_compositional_2020 nie_controllable_2021', r'\$k\$-means': 'ginart_making_2019',
    r'[Dd]ecremental|[Ii]ncremental[- ]decremental': 'hall_adding_2002', r'[Mm]atrix completion': 'xu_netflix_2023',
    r'[Rr]idge head': 'ceci_exact_2027', r'[Dd]ata redaction|redaction': 'kong_data_2023',
    r'[Ff]eature unlearning': 'moon_feature_2024', r'[Mm]achine unlearning|SISA': 'bourtoule_machine_2021 cao_towards_2015',
    r'[Cc]ontrastive divergence': 'hinton_training_2002', r'[Dd]isentangl': 'montero2021role montero_lost_2022',
    r'[Mm]ultilinear|[Bb]ilinear model': 'tenenbaum_separating_2000', r'[Ss]tyle and content': 'tenenbaum_separating_2000',
    r'[Hh]istory[- ]independen': 'blelloch_strongly_2007', r'[Ss]table[- ]matching': 'blelloch_strongly_2007',
    r'[Gg]olatkar|Achille': 'golatkar', r'[Aa]dditive decoder': 'lachapelle_additive_2023',
    r'[Pp]hysical systems|spin glass|Ising': 'hopfield_neural_1982', r'[Cc]urrent divergence|chiral|spurion': 'gell-mann_behavior_1968',
    r'\bGAN\b|\bGANs\b': '', r'\bVAEs?\b': '', r'[Dd]iffusion model': '',
    r'\bPCA\b|[Pp]rincipal[- ]component': '', r'\bMLP\b': '', r'\bRNN\b|GRU': '', r'Genie|DIAMOND|IRIS\b': '',
    r'\bCLEVR\b|dSprites|MPI3D|CelebA|MNIST': '', r'Adam\b|AdamW': '', r'[Ee]ckart|Young': '', r'[Ll]ottery': '',
}


def strip(s):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', l) for l in s.split('\n'))


src = open(sys.argv[1]).read()
lines = src.split('\n')
a0 = next(i for i, l in enumerate(lines) if l.startswith('\\end{abstract}'))
ap = next(i for i, l in enumerate(lines) if l.startswith('\\appendix'))
bib = next(i for i, l in enumerate(lines) if 'IfFileExists{references.bib}' in l or 'IfFileExists{refs_clu_iclr.bib}' in l)
body = strip(src)
blines = body.split('\n')
starts = [0]
for l in blines:
    starts.append(starts[-1] + len(l) + 1)


def lineof(pos):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        m = (lo + hi + 1) // 2
        if starts[m] <= pos:
            lo = m
        else:
            hi = m - 1
    return lo


def sentence(pos):
    a = max(body.rfind('. ', 0, pos), body.rfind('\n\n', 0, pos), body.rfind('\\item', 0, pos), body.rfind('\\caption', 0, pos)) + 1
    ends = [x for x in (body.find('. ', pos), body.find('\n\n', pos)) if x >= 0]
    return body[a:min(ends) + 1 if ends else pos + 300]


print('| term | bib key(s) | region | first line | cite in sentence | context |\n|---|---|---|---|---|---|')
for t, keys in TERMS.items():
    for region, lo, hi in (('abstract', 0, a0), ('main', a0, bib), ('appendix', ap, len(blines))):
        for m in re.finditer(t, body):
            ln = lineof(m.start())
            if lo <= ln < hi and not blines[ln].lstrip().startswith(('\\label', '\\includegraphics')):
                sen = sentence(m.start())
                c = re.findall(r'\\(?:no)?cite[a-z]*\{[^}]*\}', sen)
                ctx = ' '.join(body[max(0, m.start() - 70):m.end() + 70].split()).replace('|', '/')
                print(f'| `{t}` | {keys or "NOT IN BIB"} | {region} | {ln + 1} | {"; ".join(c) or "-"} | {ctx} |')
                break
