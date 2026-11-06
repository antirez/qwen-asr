/* Independent scalar oracles: tests odd tails, masking, cache offsets and ties. */
static void check_array(const char *name,const float *actual,const std::vector<float> &expected,float atol,float rtol=0) {
    float worst=0;
    for(size_t i=0;i<expected.size();i++) {
        float err=fabsf(actual[i]-expected[i]);worst=std::max(worst,err);
        if(!std::isfinite(actual[i])||err>atol+rtol*fabsf(expected[i])) {
            fprintf(stderr,"FAIL %s[%zu]: got %.9g expected %.9g error %.9g\n",name,i,actual[i],expected[i],err);
            throw std::runtime_error(name);
        }
    }
    printf("PASS %-28s max abs error %.6g\n",name,worst);
}
static std::vector<float> download(GPU &g,Buffer &b,size_t n) {
    std::vector<float>x(n);CUDA(cudaMemcpyAsync(x.data(),b.ptr,n*4,cudaMemcpyDeviceToHost,g.stream));CUDA(cudaStreamSynchronize(g.stream));return x;
}
// cudaDeviceSynchronize: a pageable cudaMemcpy may return before the DMA lands, and the
// kernels under test run on a non-blocking stream that does not order against it.
static void upload(Buffer &b,const std::vector<float>&v) {b.reserve(v.size()*4);CUDA(cudaMemcpy(b.ptr,v.data(),v.size()*4,cudaMemcpyHostToDevice));CUDA(cudaDeviceSynchronize());}
static float value(size_t i) {return sinf((float)i*.137f)*.7f+cosf((float)i*.023f)*.2f;}
static int run_selftests() {
    (void)device_properties();
    GPU g;CUDA(cudaStreamCreateWithFlags(&g.stream,cudaStreamNonBlocking));
    BLAS(cublasCreate(&g.blas));BLAS(cublasSetStream(g.blas,g.stream));
    BLAS(cublasSetMathMode(g.blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    // GEMV and GEMM against a double-accumulating CPU implementation.
    for(int n:{1,17}) {
        int in=256,out=41;std::vector<float>w(in*out),x(in*n),expect(n*out,0);
        for(size_t i=0;i<w.size();i++)w[i]=value(i)*.1f;
        for(size_t i=0;i<x.size();i++)x[i]=value(i+157);
        Matrix m=g.matrix(w.data(),in,out,false);Buffer dx,dy,scratch;
        upload(dx,x);dy.reserve(n*out*4);scratch.reserve(n*in*2);
        linear(&g,dy.as(),m,dx.as(),n,scratch);
        for(int t=0;t<n;t++)for(int r=0;r<out;r++) {
            double sum=0;for(int j=0;j<in;j++)sum+=(double)w[r*in+j]*x[t*in+j];expect[t*out+r]=sum;
        }
        auto got=download(g,dy,n*out);check_array(n==1?"FP16 GEMV":"FP16 GEMM (17 token tail)",got.data(),expect,.004f);
        if(n==1) {
            g.q8=true;m=g.matrix(w.data(),in,out,false,true);linear(&g,dy.as(),m,dx.as(),n,scratch);
            got=download(g,dy,n*out);check_array("Q8 GEMV",got.data(),expect,.015f);g.q8=false;
        }
    }
    for(bool rms:{false,true}) {
        int n=3,dim=896;std::vector<float>x(n*dim),w(dim),b(dim),ref(n*dim);
        for(size_t i=0;i<x.size();i++)x[i]=value(i)+2.f;
        for(int i=0;i<dim;i++){w[i]=value(i+7)+1.f;b[i]=value(i+97)*.1f;}
        Buffer dx;upload(dx,x);float *dw=g.vector(w.data(),dim),*db=g.vector(b.data(),dim);
        if(rms)norm<true><<<n,256,0,g.stream>>>(dx.as(),dx.as(),dw,nullptr,dim,1e-6f);
        else norm<false><<<n,256,0,g.stream>>>(dx.as(),dx.as(),dw,db,dim,1e-5f);
        for(int row=0;row<n;row++) {
            double mean=0,sq=0;for(int j=0;j<dim;j++){double v=x[row*dim+j];mean+=v;sq+=v*v;}
            mean=rms?0:mean/dim;double inv=1/sqrt(sq/dim-mean*mean+(rms?1e-6:1e-5));
            for(int j=0;j<dim;j++)ref[row*dim+j]=(x[row*dim+j]-mean)*inv*w[j]+(rms?0:b[j]);
        }
        auto got=download(g,dx,n*dim);check_array(rms?"RMSNorm in place":"LayerNorm in place",got.data(),ref,1e-5f);
    }
    for(bool causal:{false,true}) {
        int hd=causal?128:64,qh=4,kh=causal?2:4,n=causal?3:209,pos=148,kn=causal?pos+n:n;
        std::vector<float>q(n*qh*hd),k(kn*kh*hd),v(k.size()),ref(q.size());
        for(size_t i=0;i<q.size();i++)q[i]=value(i+5);
        for(size_t i=0;i<k.size();i++){k[i]=value(i+77);v[i]=value(i+301);}
        Buffer dq,dk,dv,dy,dp;upload(dq,q);upload(dk,k);upload(dv,v);dy.reserve(q.size()*4);dp.reserve(4);CUDA(cudaMemcpy(dp.ptr,&pos,4,cudaMemcpyHostToDevice));CUDA(cudaDeviceSynchronize());
        if(causal)attention<128,true><<<dim3(n,qh),128,0,g.stream>>>(dy.as(),dq.as(),dk.as(),dv.as(),qh,kh,n,104,dp.as<int>());
        else attention<64,false><<<dim3(n,qh),128,0,g.stream>>>(dy.as(),dq.as(),dk.as(),dv.as(),qh,kh,n,104,nullptr);
        for(int t=0;t<n;t++)for(int h=0;h<qh;h++) {
            int first=causal?0:t/104*104,last=causal?pos+t+1:std::min(first+104,n),head=h/(qh/kh);
            std::vector<double>p(last-first);double mx=-INFINITY,z=0;
            for(int s=first;s<last;s++){double dot=0;for(int d=0;d<hd;d++)dot+=(double)q[(t*qh+h)*hd+d]*k[(s*kh+head)*hd+d];p[s-first]=dot/sqrt(hd);mx=std::max(mx,p[s-first]);}
            for(double &a:p){a=exp(a-mx);z+=a;}
            for(int d=0;d<hd;d++){double sum=0;for(int s=first;s<last;s++)sum+=p[s-first]*v[(s*kh+head)*hd+d];ref[(t*qh+h)*hd+d]=sum/z;}
        }
        auto got=download(g,dy,ref.size());check_array(causal?"causal GQA, tile + KV offset":"encoder attention window/tail",got.data(),ref,3e-6f);
    }
    {
        int n=3,qh=4,kh=2,hd=128,qd=qh*hd,kd=kh*hd,stride=qd+2*kd,pos=37;
        std::vector<float>x(n*stride),qw(hd),kw(hd),rq(n*qd),rk((pos+n)*kd,0),rv(rk.size(),0);
        for(size_t i=0;i<x.size();i++)x[i]=value(i);
        for(int d=0;d<hd;d++){qw[d]=value(d)+1;kw[d]=value(d+32)+1;}
        Buffer dx,dq,dk,dv,dp;upload(dx,x);dq.reserve(rq.size()*4);upload(dk,rk);upload(dv,rv);dp.reserve(4);CUDA(cudaMemcpy(dp.ptr,&pos,4,cudaMemcpyHostToDevice));CUDA(cudaDeviceSynchronize());
        qk_rope_cache<<<dim3(n,qh+kh),128,0,g.stream>>>(dq.as(),dk.as(),dv.as(),dx.as(),g.vector(qw.data(),hd),g.vector(kw.data(),hd),qh,kh,hd,dp.as<int>(),1e-6f,1e6f);
        for(int t=0;t<n;t++)for(int h=0;h<qh+kh;h++) {
            bool isq=h<qh;int head=isq?h:h-qh,off=t*stride+(isq?0:qd)+head*hd;
            double sq=0;for(int d=0;d<hd;d++)sq+=(double)x[off+d]*x[off+d];double inv=1/sqrt(sq/hd+1e-6);
            for(int d=0;d<hd;d++) {
                int pair=d<64?d+64:d-64;double angle=(pos+t)*pow(1e6,-2.*(d%64)/hd);
                auto &weight=isq?qw:kw;double val=x[off+d]*inv*weight[d]*cos(angle)+(d<64?-1:1)*x[off+pair]*inv*weight[pair]*sin(angle);
                if(isq)rq[(t*qh+head)*hd+d]=val;
                else {rk[((pos+t)*kh+head)*hd+d]=val;rv[((pos+t)*kh+head)*hd+d]=x[t*stride+qd+kd+head*hd+d];}
            }
        }
        auto got=download(g,dq,rq.size());check_array("fused Q norm + NeoX RoPE",got.data(),rq,1e-5f);
        got=download(g,dk,rk.size());check_array("fused K norm + cache offset",got.data(),rk,1e-5f);
        got=download(g,dv,rv.size());check_array("V cache offset and prefix",got.data(),rv,0);
    }
    {
        int ci=3,hi=5,wi=7,ho=3,wo=4;std::vector<float>x(ci*hi*wi),ref(ho*wo*ci*9);
        for(size_t i=0;i<x.size();i++)x[i]=i*.125f;
        Buffer dx,dy;upload(dx,x);dy.reserve(ref.size()*2);
        im2col<<<blocks(ref.size()),256,0,g.stream>>>(dy.as<half>(),dx.as(),ci,hi,wi,ho,wo);
        std::vector<half>got(ref.size());CUDA(cudaMemcpyAsync(got.data(),dy.ptr,ref.size()*2,cudaMemcpyDeviceToHost,g.stream));CUDA(cudaStreamSynchronize(g.stream));
        std::vector<float>f(ref.size());
        for(int y=0;y<ho;y++)for(int xx=0;xx<wo;xx++)for(int c=0;c<ci;c++)for(int ky=0;ky<3;ky++)for(int kx=0;kx<3;kx++) {
            int iy=y*2+ky-1,ix=xx*2+kx-1,idx=((y*wo+xx)*ci+c)*9+ky*3+kx;
            ref[idx]=iy>=0&&iy<hi&&ix>=0&&ix<wi?x[(c*hi+iy)*wi+ix]:0;f[idx]=__half2float(got[idx]);
        }
        check_array("conv stride/padding/odd width",f.data(),ref,0);
    }
    {
        std::vector<float>x(777,-2);x[301]=x[513]=7;Buffer dx,dv,di,out;upload(dx,x);dv.reserve(4*4);di.reserve(4*4);out.reserve(4);
        argmax_stage<<<4,256,0,g.stream>>>(dv.as(),di.as<int>(),dx.as(),777);
        argmax_finish<<<1,256,0,g.stream>>>(out.as<int>(),dv.as(),di.as<int>(),4);
        int result;CUDA(cudaMemcpyAsync(&result,out.ptr,4,cudaMemcpyDeviceToHost,g.stream));CUDA(cudaStreamSynchronize(g.stream));
        if(result!=301)throw std::runtime_error("argmax tie/tail");printf("PASS argmax tie and tail\n");
    }
    {
        qwen_ctx_t ctx{};ctx.cuda=&g;ctx.config.dec_layers=1;
        ctx.config.dec_kv_heads=2;ctx.config.dec_head_dim=128;ctx.kv_cache_len=511;
        g.kv_cap=512;
        std::vector<float> keys(512*256),values(keys.size());
        for(size_t i=0;i<keys.size();i++){keys[i]=value(i);values[i]=value(i+83);}
        upload(g.dec[0].k,keys);upload(g.dec[0].v,values);
        Buffer tmp;tmp.reserve(512*256*2);
        CUDA(cudaStreamBeginCapture(g.stream,cudaStreamCaptureModeThreadLocal));
        to_half<<<blocks(keys.size()),256,0,g.stream>>>(tmp.as<half>(),g.dec[0].k.as(),keys.size());
        CUDA(cudaStreamEndCapture(g.stream,&g.graph));
        CUDA(cudaGraphInstantiate(&g.graph_exec,g.graph,0));
        ensure_kv(&ctx,700);
        if(g.kv_cap!=1024||g.graph||g.graph_exec)throw std::runtime_error("KV growth graph invalidation");
        keys.resize(511*256);values.resize(keys.size());
        auto got=download(g,g.dec[0].k,keys.size());check_array("KV growth preserves K prefix",got.data(),keys,0);
        got=download(g,g.dec[0].v,values.size());check_array("KV growth preserves V prefix",got.data(),values,0);
        puts("PASS KV growth invalidates CUDA graph");
    }
    CUDA(cudaGetLastError());return 0;
}
extern "C" int qwen_cuda_selftest(void) {
    try { return run_selftests(); }catch(const std::exception &e){fprintf(stderr,"Kernel test failed: %s\n",e.what());return 1;}
}
