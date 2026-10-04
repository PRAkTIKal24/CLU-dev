"""Industrial datasets for the F2/F3 evaluation harness.

Datasets (see each module's docstring for license + download notes):

=============  ==========================================  ========  ==========
key            what                                        labels    role
=============  ==========================================  ========  ==========
voraus_ad      6-axis robot pick&place (CC BY-NC-SA 4.0)   episode   headline
skab           water-circulation rig (GPL-3.0 — no vendor) point     iteration
tep_rieth      Tennessee Eastman, Rieth ed. (CC0)          point     scale
smd_tsb        SMD via TSB-AD curation (Apache/MIT)        point     negative
                                                                     control
mimii          acoustic machines (CC BY-SA, ~100 GB)       episode   stretch
                                                                     skeleton
=============  ==========================================  ========  ==========

Loader modules import pandas/pyarrow/pyreadr lazily — install the eval extra
(``uv sync --extra eval``) before touching voraus_ad/skab/tep_rieth/smd_tsb.
"""
