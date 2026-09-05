"""Original V2 Fig 1 generator, isolated: run make_figures.fig2_mo() only."""
import os, sys
src = open(".claude/scratch/v2-full-runs/make_figures.py").read()
src = src.replace("""if __name__ == "__main__":
    fig1_gmor()
    fig2_mo()
    fig3_gamma()
    fig4_emergent()
    fig5_isotropy()
    fig6_ep()
    fig7_collapse()
    print("ALL FIGURES DONE")""", "fig2_mo()")
g = {"__file__": os.path.abspath(".claude/scratch/v2-full-runs/make_figures.py"), "__name__": "patched"}
exec(compile(src, "make_figures.py(fig2_mo only)", "exec"), g)
