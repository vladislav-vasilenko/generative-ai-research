"""Load pinned KVAE video/image modules without importing unrelated audio stack."""
import importlib,sys,types
from .common import ROOT

def load_kvae_classes():
    root=ROOT/'reference/kvae/kvae'
    name='_kvae_course'
    for suffix,path in [('',root),('.layers',root/'layers'),('.models',root/'models')]:
        module=types.ModuleType(name+suffix); module.__path__=[str(path)]
        sys.modules.setdefault(name+suffix,module)
    video=importlib.import_module(name+'.models.kvae_3d').KVAEVideo
    image=importlib.import_module(name+'.models.kvae_2d').KVAEImage
    return video,image
