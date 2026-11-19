"""Only modules required by LGRC-Net and its supplied ablations."""
from .pcb_hybrid_backbone import LiteP5Encoder
from .pcb_c3k2_edge import C3k2_MSEdge
from .pcb_c3k2_select import C3k2_MSSelect
from .p3_local_refine import C3k2_P3LocalRefine, C3k2_P3LocalRefineNoGate
from .tri_scale_residual_fusion import TriScaleResidualFusion, ScaleFeatureSelect
from .head import Detect_LSDECD
from .p3_native_reg_head import Detect_LSDECD_P3NativeReg
from .lgrc_no_frm_head import Detect_LSDECD_NoFRM
