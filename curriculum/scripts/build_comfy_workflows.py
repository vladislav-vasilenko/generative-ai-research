"""Build model-free and VAE-only study graphs. Never queues jobs or downloads weights."""
from pathlib import Path
import copy,hashlib,json,uuid
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'curriculum/workflows';OUT.mkdir(exist_ok=True)
ASSETS=ROOT/'curriculum/assets';ASSETS.mkdir(exist_ok=True)
y,x=np.mgrid[0:256,0:256]
chart=np.stack([x/255,y/255,((x//16+y//16)%2).astype(float)],axis=-1)
Image.fromarray(np.round(chart*255).astype(np.uint8)).save(ASSETS/'curriculum_test_chart.png')

def node(i,kind,pos,inputs,outputs,widgets,title=None):
    return {'id':i,'type':kind,'pos':pos,'size':[300,150],'flags':{},'order':i-1,'mode':0,
            'inputs':[{'name':name,'type':t,'link':None} for name,t in inputs],
            'outputs':[{'name':name,'type':t,'links':[],'slot_index':j} for j,(name,t) in enumerate(outputs)],
            'properties':{'Node name for S&R':kind,'cnr_id':'comfy-core'},
            'widgets_values':widgets,**({'title':title} if title else {})}

def graph(name,nodes,specs):
    lookup={n['id']:n for n in nodes};links=[]
    for a,out_slot,b,input_name,kind in specs:
        target_slot=next(j for j,x in enumerate(lookup[b]['inputs']) if x['name']==input_name)
        link_id=len(links)+1;links.append([link_id,a,out_slot,b,target_slot,kind])
        assert lookup[a]['outputs'][out_slot]['type']==kind
        assert lookup[b]['inputs'][target_slot]['type']==kind
        lookup[a]['outputs'][out_slot]['links'].append(link_id)
        lookup[b]['inputs'][target_slot]['link']=link_id
    return {'id':str(uuid.uuid5(uuid.NAMESPACE_URL,name)),'revision':0,'last_node_id':max(lookup),
        'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],'config':{},
        'extra':{'name':name,'description':'Research teaching graph; static validation only. Upload curriculum_test_chart.png before Queue.'},
        'version':0.4}

load=lambda:node(1,'LoadImage',[0,180],[],[('IMAGE','IMAGE'),('MASK','MASK')],['curriculum_test_chart.png','image'],'Upload the supplied test chart')
save=lambda i,prefix,pos:node(i,'SaveImage',pos,[('images','IMAGE')],[('images','IMAGE')],[prefix])
echo=graph('Course C0 - image IO without models',[load(),save(2,'curriculum/c0_identity',[420,180])],[(1,0,2,'images','IMAGE')])
(OUT/'00_image_io_no_models.json').write_text(json.dumps(echo,ensure_ascii=False,indent=2)+'\n')
# The known loader/decode definitions are reused from the existing local K6 baseline.
base=json.loads((ROOT/'kandinsky-6/workflows/k6_lite_distill_smoke.json').read_text())
vae=copy.deepcopy(next(n for n in base['nodes'] if n['type']=='VAELoader'))
vae.update(id=2,pos=[0,0],order=1);vae['outputs'][0]['links']=[]
encode=node(3,'VAEEncode',[420,80],[('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')],[])
decode=node(4,'VAEDecode',[800,80],[('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')],[])
recon=graph('Course C4 - VAE reconstruction without DiT or Qwen',
    [load(),vae,encode,decode,save(5,'curriculum/c4_original',[420,360]),save(6,'curriculum/c4_reconstruction',[1170,80])],
    [(1,0,3,'pixels','IMAGE'),(2,0,3,'vae','VAE'),(3,0,4,'samples','LATENT'),
     (2,0,4,'vae','VAE'),(1,0,5,'images','IMAGE'),(4,0,6,'images','IMAGE')])
(OUT/'04_hunyuan_vae_reconstruction.json').write_text(json.dumps(recon,ensure_ascii=False,indent=2)+'\n')
print('Created:',len(echo['nodes']),'node IO graph;',len(recon['nodes']),'node VAE graph; no runtime jobs submitted')
