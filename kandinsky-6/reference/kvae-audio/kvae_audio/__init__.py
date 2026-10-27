__version__ = "1.0.0"

import audiotools

audiotools.ml.BaseModel.INTERN += ["kvae_audio.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]

from . import model, nn
from .model import KVAEAudio
