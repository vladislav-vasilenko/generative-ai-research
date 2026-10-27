"""Compare aligned RGB pixels; no hidden resize and no model inference."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
parser=argparse.ArgumentParser()
parser.add_argument('--original',required=True,type=Path)
parser.add_argument('--reconstruction',required=True,type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
def read(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'),dtype=np.float64)/255
x,y=read(args.original),read(args.reconstruction)
if x.shape!=y.shape:raise ValueError(f'Shape mismatch {x.shape} vs {y.shape}; record preprocessing instead of silently resizing.')
mse=float(np.mean((x-y)**2));exact=bool(np.array_equal(x,y))
report={'scope':'aligned RGB pixel comparison, no ICC conversion','shape':list(x.shape),
 'mse':mse,'max_absolute_error':float(np.max(np.abs(x-y))),
 'psnr_db':None if mse==0 else float(-10*np.log10(mse)),
 'exact_pixel_match':exact,'psnr_note':'infinite when MSE=0; represented as null'}
text=json.dumps(report,indent=2,allow_nan=False)+'\n'
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
print(text,end='')
