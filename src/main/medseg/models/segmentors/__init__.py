# Important that this is here, otherwise the class mapping in the builder won't function!

# If new architectures are added, the main neural net module of the architecture should be imported here
# the framework will match the name from the yaml config file (key architecture -> type) to the class name
# given here (e.g. "FCBFormer").
from .fcbformer.fcbformer import FCBFormer
from .hardnet.hardnet_dfus import HarDNetDFUS
from .hiformer.hiformer import HiFormer, HiFormerS, HiFormerB, HiFormerL
from .segformer.segformer import SegformerB0, SegformerB1, SegformerB2, SegformerB3, SegformerB4, SegformerB5, Segformer
from .segnext.segnext import SegNeXtT, SegNeXtS, SegNeXtT, SegNeXtB, SegNeXtL, SegNeXt
from .unet.unet import UNet
from .unext.unext import UNeXt, UNeXtS
