"""Exact-printed-size canvas helper for a FREELY CHOSEN aspect ratio.

canvas_wh(w_in, h_in) -> (figsize, dpi, (px_w, px_h))
The PNG is w_in x h_in at `dpi`, integer pixels.  Placed by LaTeX at
`width = w_in`, its printed height is w_in * px_h/px_w, which equals h_in to
within one pixel (<0.003 in at dpi=400).
"""
def canvas_wh(w_in, h_in, dpi=400.0):
    px_w = int(round(w_in * dpi))
    px_h = int(round(h_in * dpi))
    return (px_w / dpi, px_h / dpi), dpi, (px_w, px_h)

def printed_height(w_in, px_w, px_h):
    return w_in * px_h / px_w
