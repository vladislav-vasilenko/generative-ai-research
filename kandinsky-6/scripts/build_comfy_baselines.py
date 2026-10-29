"""Derive minimal Lite workflows from the pinned official graph.

Graph rewiring is checked here; execution still needs the user's installed
ComfyUI nodes and model files. Never describe static validation as generation.
"""
from pathlib import Path
import copy,json,uuid
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reference/kandinsky-6/comfyui/example_workflows/Kandinsky 6.0 Text to Video+Audio.json'
base=json.loads(source.read_text());out=ROOT/'workflows';out.mkdir(exist_ok=True)
for profile,width,height,frames in [('smoke',432,256,33),('baseline',864,480,121)]:
    workflow=copy.deepcopy(base)
    nodes={n['id']:n for n in workflow['nodes'] if 1<=n['id']<=12}
    nodes[1]['widgets_values'][0]='Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors'
    for entry in nodes[1].get('properties',{}).get('models',[]):
        entry['name']=entry['name'].replace('Kandinsky-6.0-Pro-distill','Kandinsky-6.0-Lite-distill')
        entry['url']=entry['url'].replace('Kandinsky-6.0-Pro-distill','Kandinsky-6.0-Lite-distill')
    nodes[1]['title']='K6 Lite Distill — original weights'
    nodes[5]['title']='Fixed video and audio captions'
    nodes[5]['inputs']=[x for x in nodes[5]['inputs'] if x['name']=='clip']
    nodes[5]['widgets_values']=[
        'A person speaks clearly towards a static camera in a quiet room. '
        '<S>Today we test the voice model.<E> Natural lighting, medium shot.',
        'Clean English speech, one speaker, no music, no background noise.']
    nodes[7]['widgets_values']=[width,height,frames,24,1]
    nodes[8]['widgets_values']=[42,'fixed',10,1,'euler','simple',1]
    nodes[9]['type']='VAEDecodeTiled'
    nodes[9]['title']='Video VAE — tiled decode'
    nodes[9]['properties']['Node name for S&R']='VAEDecodeTiled'
    nodes[9]['widgets_values']=[256,64,16,4]
    nodes[9]['size']=[300,180]
    for name in ['tile_size','overlap','temporal_size','temporal_overlap']:
        nodes[9]['inputs'].append({'name':name,'type':'INT','widget':{'name':name},'link':None})
    nodes[12]['widgets_values'][0]=f'kandinsky6_research/{profile}'
    for n in nodes.values():
        for inp in n.get('inputs',[]):inp['link']=None
        for output in n.get('outputs',[]):output['links']=[]
    links=[]
    def connect(a,slot,b,input_name,kind):
        index=next(i for i,x in enumerate(nodes[b]['inputs']) if x['name']==input_name)
        link_id=len(links)+1
        links.append([link_id,a,slot,b,index,kind])
        nodes[a]['outputs'][slot]['links'].append(link_id);nodes[b]['inputs'][index]['link']=link_id
    for spec in [
        (1,0,8,'model','MODEL'),(2,0,5,'clip','CLIP'),(2,0,6,'clip','CLIP'),
        (5,0,8,'positive','CONDITIONING'),(6,0,8,'negative','CONDITIONING'),
        (7,0,8,'latent_image','LATENT'),(8,0,9,'samples','LATENT'),(3,0,9,'vae','VAE'),
        (4,0,10,'audio_vae','VAE'),(8,0,10,'joint_latent','LATENT'),
        (9,0,11,'images','IMAGE'),(10,0,11,'audio','AUDIO'),(7,1,11,'fps','FLOAT'),
        (11,0,12,'video','VIDEO'),
    ]:connect(*spec)
    for link_id,a,slot,b,index,kind in links:
        assert nodes[a]['outputs'][slot]['type']==kind
        assert nodes[b]['inputs'][index]['type']==kind
        assert nodes[b]['inputs'][index]['link']==link_id
    workflow.update(id=str(uuid.uuid5(uuid.NAMESPACE_URL,'k6-research-'+profile)),revision=0,
                    nodes=list(nodes.values()),links=links,groups=[],last_node_id=12,last_link_id=len(links),
                    name=f'K6 Lite Distill research {profile}',
                    description='Derived from official K6 graph; no VSR, beautifier or MagCache. Static graph checks only.')
    workflow['extra']={'name':workflow['name'],'description':workflow['description']}
    serialized=json.dumps(workflow,ensure_ascii=False)
    assert 'Kandinsky-6.0-Pro' not in serialized
    assert all(n['type'] not in ['Kandinsky6VSRUpscale','Kandinsky6BeautifyPrompt','Kandinsky6MagCache'] for n in nodes.values())
    (out/f'k6_lite_distill_{profile}.json').write_text(json.dumps(workflow,ensure_ascii=False,indent=2)+'\n')
    print(profile,len(nodes),'nodes',len(links),'validated links')
