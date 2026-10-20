"""Normalize Q/K with the released FLA L2 kernel."""

from memt_sglang._vendor.fla.modules.l2norm import l2norm_fwd


def l2norm(inputs):
    output, _ = l2norm_fwd(inputs)
    return output
