"""Functional MLX port of K6's native single-modality AUDIO forward.

Preserves published tensor names and the unusual double-normalized output head.
This does NOT implement the joint PiFlow pipeline, audio VAE, vocoder or Qwen.
Passing frozen weights as a closure and differentiating only film parameters is
the proposed adapter-only ablation; it is not the voice paper's full fine-tuning.
"""
import math
import mlx.core as mx

def audio_key(key):return key.startswith('audio_') or '.audio_dec_block.' in key

class AudioCrossDiT:
    def __init__(self,weights,config):
        self.w={k:v for k,v in weights.items() if audio_key(k)}
        self.config=dict(config)
        self.width=config['model_dim_a']
        self.head_dim=sum(config['axes_dims_a'])
        self.blocks=config['num_visual_blocks']
        self.text_blocks=config['num_text_blocks']

    def linear(self,x,prefix,fp32=False):
        weight=self.w[prefix+'.weight'];bias=self.w.get(prefix+'.bias')
        if fp32:x=x.astype(mx.float32);weight=weight.astype(mx.float32);bias=None if bias is None else bias.astype(mx.float32)
        else:x=x.astype(weight.dtype)
        out=x@weight.T
        return out if bias is None else out+bias

    def norm(self,x,prefix=None):
        dtype=x.dtype;work=x.astype(mx.float32)
        mean=mx.mean(work,axis=-1,keepdims=True)
        var=mx.mean(mx.square(work-mean),axis=-1,keepdims=True)
        out=(work-mean)*mx.rsqrt(var+1e-5)
        if prefix is not None:
            out=out*self.w[prefix+'.weight'].astype(mx.float32)+self.w[prefix+'.bias'].astype(mx.float32)
        return out.astype(dtype)

    def embed(self,x,prefix):return self.norm(self.linear(x,prefix+'.in_layer'),prefix+'.norm')

    def rms(self,x,prefix):
        # torch.nn.RMSNorm(eps=None) uses the input dtype's machine epsilon.
        eps=mx.finfo(x.dtype).eps
        work=x.astype(mx.float32)
        return (work*mx.rsqrt(mx.mean(mx.square(work),axis=-1,keepdims=True)+eps)).astype(x.dtype)*self.w[prefix+'.weight']

    def rope(self,length,scale=1.):
        frequencies=mx.exp(-math.log(10000.)*mx.arange(self.head_dim//2,dtype=mx.float32)/(self.head_dim//2))*scale
        args=mx.arange(length,dtype=mx.float32)[:,None]*frequencies[None,:]
        return mx.cos(args)[None,:,None,:],mx.sin(args)[None,:,None,:]

    @staticmethod
    def rotate(x,rope):
        dtype=x.dtype;shape=x.shape
        pair=x.astype(mx.float32).reshape(*shape[:-1],shape[-1]//2,2)
        even,odd=pair[...,0],pair[...,1];c,s=rope
        return mx.stack([c*even-s*odd,s*even+c*odd],axis=-1).reshape(shape).astype(dtype)

    def attention(self,x,context,prefix,rope=None,mask=None):
        heads=x.shape[-1]//self.head_dim
        q=self.linear(x,prefix+'.to_query').reshape(x.shape[0],x.shape[1],heads,self.head_dim)
        k=self.linear(context,prefix+'.to_key').reshape(context.shape[0],context.shape[1],heads,self.head_dim)
        v=self.linear(context,prefix+'.to_value').reshape(context.shape[0],context.shape[1],heads,self.head_dim)
        q=self.rms(q,prefix+'.query_norm');k=self.rms(k,prefix+'.key_norm')
        if rope is not None:q=self.rotate(q,rope);k=self.rotate(k,rope)
        q,k,v=(a.transpose(0,2,1,3) for a in (q,k,v))
        out=mx.fast.scaled_dot_product_attention(q,k,v,scale=self.head_dim**-0.5,mask=mask)
        return self.linear(out.transpose(0,2,1,3).reshape(x.shape),prefix+'.out_layer')

    def feed_forward(self,x,prefix):
        x=self.linear(x,prefix+'.net.0.proj')
        x=0.5*x*(1+mx.erf(x/math.sqrt(2.)))  # exact GELU, no tanh approximation
        return self.linear(x,prefix+'.net.2')

    def modulation(self,time,prefix):
        work=time.astype(mx.float32)
        return self.linear(work*mx.sigmoid(work),prefix+'.out_layer',fp32=True).astype(time.dtype)

    def affine_norm(self,x,scale,shift):
        return (self.norm(x).astype(mx.float32)*(1+scale[:,None,:].astype(mx.float32))+shift[:,None,:].astype(mx.float32)).astype(x.dtype)

    @staticmethod
    def residual(x,update,gate):
        return (x.astype(mx.float32)+gate[:,None,:].astype(mx.float32)*update.astype(mx.float32)).astype(x.dtype)

    def time_embedding(self,time,pooled):
        width=self.width
        freq=mx.exp(-math.log(10000.)*mx.arange(width//2,dtype=mx.float32)/(width//2))
        args=time.astype(mx.float32)[:,None]*freq[None,:]
        z=mx.concatenate([mx.cos(args),mx.sin(args)],axis=-1)
        prefix='audio_time_embeddings.timestep_embedder'
        z=self.linear(z,prefix+'.linear_1',fp32=True)
        z=self.linear(z*mx.sigmoid(z),prefix+'.linear_2',fp32=True)
        dtype=self.w[prefix+'.linear_2.weight'].dtype
        return z.astype(dtype)+self.embed(pooled,'audio_pooled_text_embeddings')

    def __call__(self,x,text,pooled,time,mask=None,film=None,speaker=None,reference_tokens=0):
        if x.ndim!=3 or text.ndim!=3:raise ValueError('Expected [B,L,C] latents and text')
        if mask is not None and mask.ndim==2:mask=mask[:,None,None,:]
        tm=self.time_embedding(time,pooled)
        text=self.embed(text,'audio_text_embeddings')
        text_rope=self.rope(text.shape[1])
        for i in range(self.text_blocks):
            p=f'audio_text_transformer_blocks.{i}'
            sa,ff=mx.split(self.modulation(tm,p+'.text_modulation'),2,axis=-1)
            shift,scale,gate=mx.split(sa,3,axis=-1)
            z=self.affine_norm(text,scale,shift)
            text=self.residual(text,self.attention(z,z,p+'.attn',text_rope,mask),gate)
            shift,scale,gate=mx.split(ff,3,axis=-1)
            text=self.residual(text,self.feed_forward(self.affine_norm(text,scale,shift),p+'.feed_forward'),gate)
        hidden=self.embed(x,'audio_embeddings')
        if film is not None and speaker is not None:
            coeff=speaker@film['weight'].T+film['bias']
            gamma,beta=mx.split(coeff,2,axis=-1)
            target_mask=(mx.arange(hidden.shape[1])>=reference_tokens)[None,:,None]
            hidden=hidden*(1+gamma[:,None,:]*target_mask)+beta[:,None,:]*target_mask
        audio_rope=self.rope(hidden.shape[1],self.config.get('audio_freqs_scaling',1.))
        for i in range(self.blocks):
            p=f'visual_transformer_blocks.{i}.audio_dec_block'
            sa,ca,ff=mx.split(self.modulation(tm,p+'.visual_modulation'),3,axis=-1)
            shift,scale,gate=mx.split(sa,3,axis=-1)
            z=self.affine_norm(hidden,scale,shift)
            hidden=self.residual(hidden,self.attention(z,z,p+'.self_attention',audio_rope),gate)
            shift,scale,gate=mx.split(ca,3,axis=-1)
            z=self.affine_norm(hidden,scale,shift)
            hidden=self.residual(hidden,self.attention(z,text,p+'.cross_attention',mask=mask),gate)
            shift,scale,gate=mx.split(ff,3,axis=-1)
            hidden=self.residual(hidden,self.feed_forward(self.affine_norm(hidden,scale,shift),p+'.feed_forward'),gate)
        shift,scale=mx.split(self.modulation(tm,'audio_out_layer.modulation'),2,axis=-1)
        hidden=self.affine_norm(hidden,scale,shift)
        hidden=self.norm(hidden)  # preserve native K6's second output-head LayerNorm
        return self.linear(hidden,'audio_out_layer.out_layer')

    @staticmethod
    def as_policy_grid(output,latent_channels=40):
        b,length,channels=output.shape
        if channels%latent_channels:raise ValueError('Policy output does not divide into latent channels')
        return output.reshape(b,length,channels//latent_channels,latent_channels).transpose(0,2,1,3)
