/* Native CUDA execution for Qwen3-ASR. No Python/ONNX/torch in inference. */
extern "C" {
#include "qwen_asr_cuda.h"
}
#include <cublas_v2.h>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda/kernels.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"CUDA: %s at %s:%d: %s\n",#call,__FILE__,__LINE__,cudaGetErrorString(e)); exit(EXIT_FAILURE); }} while(0)
#define BLAS(call) do { cublasStatus_t e=(call); if(e!=CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS: %s failed (%d) at %s:%d\n",#call,(int)e,__FILE__,__LINE__); exit(EXIT_FAILURE); }} while(0)
static int blocks(size_t n) { return (int)((n+255)/256); }
static cudaDeviceProp device_properties() {
    int device=0; cudaDeviceProp p;
    CUDA(cudaGetDevice(&device)); CUDA(cudaGetDeviceProperties(&p,device));
    if(p.warpSize!=32)
        throw std::runtime_error("this CUDA backend requires 32-thread warps");
    return p;
}
struct Buffer {
    void *ptr=nullptr; size_t bytes=0;
    Buffer()=default; Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
    ~Buffer() { if(ptr) (void)cudaFree(ptr); }
    void reserve(size_t n) {
        if(n<=bytes) return;
        if(ptr) CUDA(cudaFree(ptr));
        CUDA(cudaMalloc(&ptr,n)); bytes=n;
    }
    template<class T=float> T *as() { return (T*)ptr; }
};
struct Matrix { half *w=nullptr; int8_t *q8=nullptr; float *scale=nullptr; int in=0,out=0; };
struct EncLayer { Matrix qkv,o,fc1,fc2; float *qkvb,*ob,*b1,*b2,*nw1,*nb1,*nw2,*nb2; };
struct DecLayer { Matrix qkv,o,gu,down; float *qn,*kn,*n1,*n2; Buffer k,v; };
struct Workspace {
    Buffer x,xn,qkv,q,k,v,a,p,ff,gu,h;
    void reserve(int n,int dim,int qdim,int kvdim,int inter) {
        x.reserve((size_t)n*dim*4); xn.reserve((size_t)n*dim*4);
        qkv.reserve((size_t)n*(qdim+2*kvdim)*4);
        q.reserve((size_t)n*qdim*4);k.reserve((size_t)n*kvdim*4);v.reserve((size_t)n*kvdim*4);
        a.reserve((size_t)n*qdim*4);p.reserve((size_t)n*dim*4);
        ff.reserve((size_t)n*inter*4);gu.reserve((size_t)n*2*inter*4);
        h.reserve((size_t)n*std::max({dim,qdim,inter,7680})*2);
    }
};
struct GPU {
    cudaStream_t stream=nullptr; cublasHandle_t blas=nullptr;
    cudaGraph_t graph=nullptr; cudaGraphExec_t graph_exec=nullptr;
    std::vector<void*> weights;
    EncLayer enc[QWEN_MAX_ENC_LAYERS]; DecLayer dec[QWEN_MAX_DEC_LAYERS];
    Matrix conv[3],conv_out,proj1,proj2,embedding;
    float *convb[3],*postw,*postb,*proj1b,*proj2b,*final_norm;
    Buffer position,result,logits,maxv,maxi;
    Buffer col,convx,convy,convt,reshape;
    Workspace encoder,prefill,decode;
    int kv_cap=0; bool q8=false,graphs=true;
    size_t weight_bytes=0;
    ~GPU() {
        if(stream) (void)cudaStreamSynchronize(stream);
        invalidate_graph();
        for(void *p:weights) (void)cudaFree(p);
        if(blas) cublasDestroy(blas);
        if(stream) (void)cudaStreamDestroy(stream);
    }
    void invalidate_graph() {
        if(graph_exec) CUDA(cudaGraphExecDestroy(graph_exec));
        if(graph) CUDA(cudaGraphDestroy(graph));
        graph_exec=nullptr;graph=nullptr;
    }
    template<class T> T *alloc(size_t n) {
        T *p; CUDA(cudaMalloc((void**)&p,n*sizeof(T)));
        weights.push_back(p);weight_bytes+=n*sizeof(T); return p;
    }
    float *vector(const float *x,int n) {
        if(!x) throw std::runtime_error("missing weight vector");
        // Copy on the work stream and wait: a pageable cudaMemcpy may return before the DMA lands,
        // and the non-blocking stream does not order against the legacy default stream.
        float *p=alloc<float>(n); CUDA(cudaMemcpyAsync(p,x,n*4,cudaMemcpyHostToDevice,stream));
        CUDA(cudaStreamSynchronize(stream));return p;
    }
    Matrix matrix(const void *x,int in,int out,bool bf16,bool quant=false) {
        if(!x) throw std::runtime_error("missing weight matrix");
        Matrix m; m.in=in;m.out=out;size_t n=(size_t)in*out;
        m.w=alloc<half>(n);
        Buffer tmp;tmp.reserve(n*(bf16?2:4));
        // Same stream as the conversion kernel below, so it cannot read a partly copied buffer.
        CUDA(cudaMemcpyAsync(tmp.ptr,x,tmp.bytes,cudaMemcpyHostToDevice,stream));
        if(bf16) bf16_to_half<<<blocks(n),256,0,stream>>>(m.w,tmp.as<uint16_t>(),n);
        else to_half<<<blocks(n),256,0,stream>>>(m.w,tmp.as(),n);
        if(quant && q8) {
            m.q8=alloc<int8_t>(n);m.scale=alloc<float>(n/32);
            quantize_q8<<<(n/32+7)/8,256,0,stream>>>(m.q8,m.scale,m.w,n/32);
        }
        CUDA(cudaStreamSynchronize(stream));return m;
    }
    Matrix fused(const void *a,const void *b,const void *c,int in,int na,int nb,int nc,bool bf16,bool quant) {
        if(!a||!b||(nc&&!c)) throw std::runtime_error("missing fused matrix");
        size_t rowbytes=(size_t)in*(bf16?2:4);
        std::vector<unsigned char> tmp((na+nb+nc)*rowbytes);
        memcpy(tmp.data(),a,na*rowbytes);memcpy(tmp.data()+na*rowbytes,b,nb*rowbytes);
        if(nc) memcpy(tmp.data()+(na+nb)*rowbytes,c,nc*rowbytes);
        return matrix(tmp.data(),in,na+nb+nc,bf16,quant);
    }
};
static GPU *gpu(qwen_ctx_t *ctx) { return (GPU*)ctx->cuda; }
static void gemm_half(GPU *g,float *y,const Matrix &w,const half *xh,int n) {
    float alpha=1,beta=0;
    BLAS(cublasGemmEx(g->blas,CUBLAS_OP_T,CUBLAS_OP_N,
        w.out,n,w.in,&alpha,w.w,CUDA_R_16F,w.in,xh,CUDA_R_16F,w.in,
        &beta,y,CUDA_R_32F,w.out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
}
static void linear(GPU *g,float *y,const Matrix &w,const float *x,int n,Buffer &scratch) {
    if(n==1) {
        if(w.q8) gemv4<true><<<(w.out+7)/8,256,w.in*4,g->stream>>>(y,w.w,w.q8,w.scale,x,w.in,w.out);
        else gemv4<false><<<(w.out+7)/8,256,w.in*4,g->stream>>>(y,w.w,nullptr,nullptr,x,w.in,w.out);
    } else {
        size_t count=(size_t)n*w.in;
        // The caller reserves the workspace outside graph capture.
        to_half<<<blocks(count),256,0,g->stream>>>(scratch.as<half>(),x,count);
        gemm_half(g,y,w,scratch.as<half>(),n);
    }
}
static void epi(GPU *g,float *y,const float *b,const float *r,int n,int dim,int act=0) {
    epilogue<<<blocks((size_t)n*dim),256,0,g->stream>>>(y,b,r,n*dim,dim,act);
}
static void ensure_kv(qwen_ctx_t *ctx,int needed) {
    GPU *g=gpu(ctx);if(needed<=g->kv_cap)return;
    g->invalidate_graph();
    CUDA(cudaStreamSynchronize(g->stream));
    int cap=std::max(512,g->kv_cap);
    while(cap<needed) { if(cap>65536)throw std::runtime_error("decoder context exceeds 131072 tokens"); cap*=2; }
    int kv=ctx->config.dec_kv_heads*ctx->config.dec_head_dim;
    size_t size=(size_t)cap*kv*4,old=(size_t)ctx->kv_cache_len*kv*4;
    for(int i=0;i<ctx->config.dec_layers;i++) for(Buffer *b:{&g->dec[i].k,&g->dec[i].v}) {
        void *p;CUDA(cudaMalloc(&p,size));
        if(b->ptr) { if(old)CUDA(cudaMemcpyAsync(p,b->ptr,old,cudaMemcpyDeviceToDevice,g->stream));CUDA(cudaStreamSynchronize(g->stream));CUDA(cudaFree(b->ptr)); }
        b->ptr=p;b->bytes=size;
    }
    g->kv_cap=cap;
}
extern "C" int qwen_cuda_init(qwen_ctx_t *ctx) {
    GPU *g=new GPU;ctx->cuda=g;
    try {
        cudaDeviceProp p=device_properties();
        CUDA(cudaStreamCreateWithFlags(&g->stream,cudaStreamNonBlocking));
        BLAS(cublasCreate(&g->blas));BLAS(cublasSetStream(g->blas,g->stream));
        BLAS(cublasSetMathMode(g->blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        const char *q8=getenv("QWEN_CUDA_Q8"),*graphs=getenv("QWEN_CUDA_GRAPHS");
        g->q8=!q8||atoi(q8)!=0;g->graphs=!graphs||atoi(graphs)!=0;
        if(qwen_verbose)fprintf(stderr,"CUDA: %s (sm_%d%d), %s decode, CUDA graphs %s\n",p.name,p.major,p.minor,g->q8?"Q8":"FP16",g->graphs?"on":"off");
        const qwen_config_t &c=ctx->config;qwen_encoder_t &e=ctx->encoder;
        int d=c.enc_d_model,f=c.enc_ffn_dim;
        const float *cw[]={e.conv1_weight,e.conv2_weight,e.conv3_weight};
        const float *cb[]={e.conv1_bias,e.conv2_bias,e.conv3_bias};
        for(int i=0;i<3;i++){g->conv[i]=g->matrix(cw[i],i?480*9:9,480,false);g->convb[i]=g->vector(cb[i],480);}
        g->conv_out=g->matrix(e.conv_out_weight,7680,d,false);
        for(int i=0;i<c.enc_layers;i++) {
            auto &s=e.layers[i];auto &l=g->enc[i];
            l.qkv=g->fused(s.wq_weight,s.wk_weight,s.wv_weight,d,d,d,d,false,false);
            std::vector<float>b(3*d);memcpy(b.data(),s.wq_bias,d*4);memcpy(b.data()+d,s.wk_bias,d*4);memcpy(b.data()+2*d,s.wv_bias,d*4);
            l.qkvb=g->vector(b.data(),3*d);l.o=g->matrix(s.wo_weight,d,d,false);l.ob=g->vector(s.wo_bias,d);
            l.fc1=g->matrix(s.fc1_weight,d,f,false);l.b1=g->vector(s.fc1_bias,f);
            l.fc2=g->matrix(s.fc2_weight,f,d,false);l.b2=g->vector(s.fc2_bias,d);
            l.nw1=g->vector(s.attn_norm_weight,d);l.nb1=g->vector(s.attn_norm_bias,d);
            l.nw2=g->vector(s.ffn_norm_weight,d);l.nb2=g->vector(s.ffn_norm_bias,d);
        }
        g->postw=g->vector(e.ln_post_weight,d);g->postb=g->vector(e.ln_post_bias,d);
        g->proj1=g->matrix(e.proj1_weight,d,d,false);g->proj1b=g->vector(e.proj1_bias,d);
        g->proj2=g->matrix(e.proj2_weight,d,c.enc_output_dim,false);g->proj2b=g->vector(e.proj2_bias,c.enc_output_dim);
        int h=c.dec_hidden,q=c.dec_heads*c.dec_head_dim,kv=c.dec_kv_heads*c.dec_head_dim,inter=c.dec_intermediate;
        for(int i=0;i<c.dec_layers;i++) {
            auto &s=ctx->decoder.layers[i];auto &l=g->dec[i];
            l.qkv=g->fused(s.wq_weight_bf16,s.wk_weight_bf16,s.wv_weight_bf16,h,q,kv,kv,true,true);
            l.o=g->matrix(s.wo_weight_bf16,q,h,true,true);
            l.gu=g->fused(s.gate_weight_bf16,s.up_weight_bf16,nullptr,h,inter,inter,0,true,true);
            l.down=g->matrix(s.down_weight_bf16,inter,h,true,true);
            l.qn=g->vector(s.q_norm_weight,c.dec_head_dim);l.kn=g->vector(s.k_norm_weight,c.dec_head_dim);
            l.n1=g->vector(s.input_norm,h);l.n2=g->vector(s.post_attn_norm,h);
        }
        g->embedding=g->matrix(ctx->decoder.tok_embeddings_bf16,h,c.vocab_size,true,true);
        g->final_norm=g->vector(ctx->decoder.norm,h);
        g->position.reserve(4);g->result.reserve(4);g->logits.reserve((size_t)c.vocab_size*4);
        g->maxv.reserve(blocks(c.vocab_size)*4);g->maxi.reserve(blocks(c.vocab_size)*4);
        g->decode.reserve(1,h,q,kv,inter);
        ensure_kv(ctx,512);
        if(qwen_verbose)fprintf(stderr,"GPU weights: %.2f GiB\n",g->weight_bytes/1073741824.);
        return 0;
    }catch(const std::exception &e){fprintf(stderr,"CUDA initialization: %s\n",e.what());return -1;}
}
extern "C" void qwen_cuda_free(qwen_ctx_t *ctx) { delete gpu(ctx);ctx->cuda=nullptr; }

extern "C" float *qwen_encoder_forward(qwen_ctx_t *ctx,const float *mel,int frames,int *out_seq) {
    GPU *g=gpu(ctx);const auto &c=ctx->config;
    if(frames<=0){*out_seq=0;return nullptr;}
    int chunk=c.enc_chunk_size,nt=0;
    for(int off=0;off<frames;off+=chunk) nt+=(std::min(chunk,frames-off)+7)/8;
    int d=c.enc_d_model;Workspace &w=g->encoder;
    w.reserve(nt,std::max(d,c.enc_output_dim),d,d,c.enc_ffn_dim);
    g->convx.reserve(480*64*((chunk+1)/2)*4);g->convy.reserve(g->convx.bytes);g->convt.reserve(g->convx.bytes);
    g->col.reserve((size_t)32*((chunk+3)/4)*480*9*2);
    g->reshape.reserve((size_t)((chunk+7)/8)*7680*4);
    std::vector<float> cm(128*chunk);int token=0;
    for(int off=0;off<frames;off+=chunk) {
        int width=std::min(chunk,frames-off);
        for(int m=0;m<128;m++)memcpy(cm.data()+m*width,mel+(size_t)m*frames+off,width*4);
        CUDA(cudaMemcpyAsync(g->convx.ptr,cm.data(),128*width*4,cudaMemcpyHostToDevice,g->stream));
        // The host staging vector is reused on the next chunk only after synchronization.
        float *x=g->convx.as(),*y=g->convy.as();int hi=128,wi=width,ci=1;
        for(int layer=0;layer<3;layer++) {
            int ho=(hi+1)/2,wo=(wi+1)/2;
            size_t nel=(size_t)ho*wo*ci*9;
            im2col<<<blocks(nel),256,0,g->stream>>>(g->col.as<half>(),x,ci,hi,wi,ho,wo);
            gemm_half(g,g->convt.as(),g->conv[layer],g->col.as<half>(),ho*wo);
            conv_finish<<<blocks(480*ho*wo),256,0,g->stream>>>(y,g->convt.as(),g->convb[layer],480,ho*wo);
            std::swap(x,y);ci=480;hi=ho;wi=wo;
        }
        conv_reshape<<<blocks(wi*7680),256,0,g->stream>>>(g->reshape.as(),x,wi,7680);
        linear(g,w.x.as()+(size_t)token*d,g->conv_out,g->reshape.as(),wi,w.h);
        position_add<<<blocks(wi*d),256,0,g->stream>>>(w.x.as()+(size_t)token*d,wi,d);
        token+=wi;
        CUDA(cudaStreamSynchronize(g->stream));
    }
    float *x=w.x.as(),*p=w.p.as();
    int window=((chunk+7)/8)*(c.enc_n_window_infer/chunk);
    for(int i=0;i<c.enc_layers;i++) {
        auto &l=g->enc[i];
        norm<false><<<nt,256,0,g->stream>>>(w.xn.as(),x,l.nw1,l.nb1,d,1e-5f);
        linear(g,w.qkv.as(),l.qkv,w.xn.as(),nt,w.h);
        split_qkv<<<blocks(nt*d),256,0,g->stream>>>(w.q.as(),w.k.as(),w.v.as(),w.qkv.as(),l.qkvb,nt,d);
        attention<64,false><<<dim3(nt,c.enc_heads),128,0,g->stream>>>(w.a.as(),w.q.as(),w.k.as(),w.v.as(),c.enc_heads,c.enc_heads,nt,window,nullptr);
        linear(g,p,l.o,w.a.as(),nt,w.h);epi(g,p,l.ob,x,nt,d);std::swap(x,p);
        norm<false><<<nt,256,0,g->stream>>>(w.xn.as(),x,l.nw2,l.nb2,d,1e-5f);
        linear(g,w.ff.as(),l.fc1,w.xn.as(),nt,w.h);epi(g,w.ff.as(),l.b1,nullptr,nt,c.enc_ffn_dim,1);
        linear(g,p,l.fc2,w.ff.as(),nt,w.h);epi(g,p,l.b2,x,nt,d);std::swap(x,p);
    }
    norm<false><<<nt,256,0,g->stream>>>(w.xn.as(),x,g->postw,g->postb,d,1e-5f);
    linear(g,w.p.as(),g->proj1,w.xn.as(),nt,w.h);epi(g,w.p.as(),g->proj1b,nullptr,nt,d,1);
    linear(g,w.x.as(),g->proj2,w.p.as(),nt,w.h);epi(g,w.x.as(),g->proj2b,nullptr,nt,c.enc_output_dim);
    size_t bytes=(size_t)nt*c.enc_output_dim*4;
    float *result=(float*)malloc(bytes);if(!result)return nullptr;
    CUDA(cudaMemcpyAsync(result,w.x.ptr,bytes,cudaMemcpyDeviceToHost,g->stream));CUDA(cudaStreamSynchronize(g->stream));
    *out_seq=nt;return result;
}
static void decoder_run(qwen_ctx_t *ctx,Workspace &w,int n,bool logits) {
    GPU *g=gpu(ctx);const auto &c=ctx->config;
    int h=c.dec_hidden,inter=c.dec_intermediate;
    float *x=w.x.as(),*p=w.p.as();
    for(int i=0;i<c.dec_layers;i++) {
        auto &l=g->dec[i];
        norm<true><<<n,256,0,g->stream>>>(w.xn.as(),x,l.n1,nullptr,h,c.dec_rms_norm_eps);
        linear(g,w.qkv.as(),l.qkv,w.xn.as(),n,w.h);
        qk_rope_cache<<<dim3(n,c.dec_heads+c.dec_kv_heads),128,0,g->stream>>>(w.q.as(),l.k.as(),l.v.as(),w.qkv.as(),l.qn,l.kn,c.dec_heads,c.dec_kv_heads,c.dec_head_dim,g->position.as<int>(),c.dec_rms_norm_eps,c.dec_rope_theta);
        attention<128,true><<<dim3(n,c.dec_heads),128,0,g->stream>>>(w.a.as(),w.q.as(),l.k.as(),l.v.as(),c.dec_heads,c.dec_kv_heads,0,0,g->position.as<int>());
        linear(g,p,l.o,w.a.as(),n,w.h);epi(g,p,nullptr,x,n,h);std::swap(x,p);
        norm<true><<<n,256,0,g->stream>>>(w.xn.as(),x,l.n2,nullptr,h,c.dec_rms_norm_eps);
        linear(g,w.gu.as(),l.gu,w.xn.as(),n,w.h);
        swiglu<<<blocks(n*inter),256,0,g->stream>>>(w.ff.as(),w.gu.as(),n*inter,inter);
        linear(g,p,l.down,w.ff.as(),n,w.h);epi(g,p,nullptr,x,n,h);std::swap(x,p);
    }
    if(logits) {
        norm<true><<<1,256,0,g->stream>>>(w.xn.as(),x,g->final_norm,nullptr,h,c.dec_rms_norm_eps);
        linear(g,g->logits.as(),g->embedding,w.xn.as(),1,w.h);
        argmax_stage<<<blocks(c.vocab_size),256,0,g->stream>>>(g->maxv.as(),g->maxi.as<int>(),g->logits.as(),c.vocab_size);
        argmax_finish<<<1,256,0,g->stream>>>(g->result.as<int>(),g->maxv.as(),g->maxi.as<int>(),blocks(c.vocab_size));
    }
}
extern "C" void qwen_decoder_prefill(qwen_ctx_t *ctx,const float *embeds,int n) {
    if(n<=0)return;
    GPU *g=gpu(ctx);const auto &c=ctx->config;
    ensure_kv(ctx,ctx->kv_cache_len+n);
    // Bound scratch allocations for long offline recordings without changing attention.
    for(int off=0;off<n;off+=256) {
        int count=std::min(256,n-off);
        g->prefill.reserve(count,c.dec_hidden,c.dec_heads*c.dec_head_dim,c.dec_kv_heads*c.dec_head_dim,c.dec_intermediate);
        CUDA(cudaMemcpyAsync(g->position.ptr,&ctx->kv_cache_len,4,cudaMemcpyHostToDevice,g->stream));
        CUDA(cudaMemcpyAsync(g->prefill.x.ptr,embeds+(size_t)off*c.dec_hidden,(size_t)count*c.dec_hidden*4,cudaMemcpyHostToDevice,g->stream));
        decoder_run(ctx,g->prefill,count,false);
        CUDA(cudaStreamSynchronize(g->stream));ctx->kv_cache_len+=count;
    }
}
extern "C" int qwen_decoder_forward(qwen_ctx_t *ctx,const float *embed) {
    GPU *g=gpu(ctx);ensure_kv(ctx,ctx->kv_cache_len+1);
    CUDA(cudaMemcpyAsync(g->position.ptr,&ctx->kv_cache_len,4,cudaMemcpyHostToDevice,g->stream));
    CUDA(cudaMemcpyAsync(g->decode.x.ptr,embed,ctx->config.dec_hidden*4,cudaMemcpyHostToDevice,g->stream));
    if(g->graphs) {
        if(!g->graph_exec) {
            CUDA(cudaStreamSynchronize(g->stream));
            CUDA(cudaStreamBeginCapture(g->stream,cudaStreamCaptureModeThreadLocal));
            decoder_run(ctx,g->decode,1,true);
            CUDA(cudaStreamEndCapture(g->stream,&g->graph));
            CUDA(cudaGraphInstantiate(&g->graph_exec,g->graph,0));
        }
        CUDA(cudaGraphLaunch(g->graph_exec,g->stream));
    }else decoder_run(ctx,g->decode,1,true);
    int result;
    CUDA(cudaMemcpyAsync(&result,g->result.ptr,4,cudaMemcpyDeviceToHost,g->stream));
    CUDA(cudaStreamSynchronize(g->stream));ctx->kv_cache_len++;
    return result;
}

#ifdef QWEN_CUDA_TEST
#include "tests/cuda/selftest.h"
#endif
