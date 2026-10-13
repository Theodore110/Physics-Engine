#include <vector>
#include <cstdint>
#include <cmath>
struct Body { float x,y,vx,vy,inv_m,r; };
struct Pair { uint32_t a,b; };
struct Contact { uint32_t a,b; float nx,ny,bias,jn; };
static uint32_t st=12345;
static inline float rnd(){ st=st*1664525u+1013904223u; return (float)(st>>8)/16777216.0f; }
extern "C" float k_run(int n,int steps){
  st=12345;
  const float dt=1.0f/60.0f,g=-9.8f,BETA=0.2f,SLOP=0.01f; const int ITERS=10;
  std::vector<Body> bs((size_t)n*n); std::vector<Pair> ps;
  for(int j=0;j<n;j++)for(int i=0;i<n;i++){
    float rx=rnd(),ry=rnd(),rv=rnd(); Body&b=bs[(size_t)j*n+i];
    b.x=(float)i+(rx-0.5f)*0.1f; b.y=(float)j+(ry-0.5f)*0.1f;
    b.vx=j==0?0.0f:(rv-0.5f)*0.2f; b.vy=0; b.inv_m=j==0?0.0f:1.0f; b.r=0.55f; }
  const int di[4]={1,0,1,-1}, dj[4]={0,1,1,1};
  for(int j=0;j<n;j++)for(int i=0;i<n;i++)for(int k=0;k<4;k++){
    int ii=i+di[k],jj=j+dj[k]; if(ii<0||ii>=n||jj>=n) continue;
    if(j==0&&jj==0) continue; ps.push_back({(uint32_t)(j*n+i),(uint32_t)(jj*n+ii)}); }
  std::vector<Contact> cs(ps.size());
  for(int s=0;s<steps;s++){
    for(auto&b:bs) if(b.inv_m>0.0f) b.vy+=g*dt;
    size_t nc=0;
    for(const Pair&p:ps){
      const Body&A=bs[p.a];const Body&B=bs[p.b];
      float dx=B.x-A.x,dy=B.y-A.y,d2=dx*dx+dy*dy,rr=A.r+B.r;
      if(d2<rr*rr&&d2>1e-12f){
        float d=std::sqrt(d2),inv=1.0f/d,pen=rr-d-SLOP; if(pen<0.0f)pen=0.0f;
        cs[nc++]={p.a,p.b,dx*inv,dy*inv,(BETA/dt)*pen,0.0f}; } }
    for(int it=0;it<ITERS;it++) for(size_t c=0;c<nc;c++){
      Contact&C=cs[c]; Body&A=bs[C.a]; Body&B=bs[C.b];
      float ia=A.inv_m,ib=B.inv_m;
      float vn=(B.vx-A.vx)*C.nx+(B.vy-A.vy)*C.ny;
      float lam=(C.bias-vn)/(ia+ib); float nj=C.jn+lam; if(nj<0.0f)nj=0.0f;
      lam=nj-C.jn; C.jn=nj; float px=lam*C.nx,py=lam*C.ny;
      A.vx-=px*ia;A.vy-=py*ia;B.vx+=px*ib;B.vy+=py*ib; }
    for(auto&b:bs){b.x+=b.vx*dt;b.y+=b.vy*dt;} }
  double sum=0; for(auto&b:bs) sum+=(double)b.x+(double)b.y; return (float)sum;
}
