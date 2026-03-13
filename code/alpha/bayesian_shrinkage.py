
def shrink(alpha, prior_mean=0.0, strength=0.5):
    shrunk={}
    for k,v in alpha.items():
        shrunk[k] = (1-strength)*v + strength*prior_mean
    return shrunk
