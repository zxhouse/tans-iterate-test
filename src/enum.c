// exhaustive enumeration of all tANS spreads for n=3 symbols and L=16 states
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double exact_rate(const int*sym,int L,int n,const double*p,double*pi,int warm); extern double TOL;
int main(int argc,char**argv){ double p[3]={atof(argv[1]),atof(argv[2]),atof(argv[3])}; int L=16,sym[16]; double pi[16];
  double best=1e9; long nbest=0, total=0; int arg[16]; long N=1; for(int i=0;i<L;i++) N*=3;
  double H=0; for(int s=0;s<3;s++) H-=p[s]*log2(p[s]);
  long near[4]={0,0,0,0}; TOL=1e-10; float*all=malloc(N*sizeof(float)); long lim=argc>4?atol(argv[4]):N; N=lim;
  for(long c=0;c<N;c++){ all[c]=-1; long v=c; int cnt[3]={0,0,0}; for(int i=0;i<L;i++){ sym[i]=v%3; cnt[sym[i]]++; v/=3; }
    if(!cnt[0]||!cnt[1]||!cnt[2]) continue; double r=exact_rate(sym,L,3,p,pi,0);
    all[c]=(float)(r-H); { total++; if(r<best-1e-9){ best=r; nbest=1; for(int i=0;i<L;i++) arg[i]=sym[i]; } else if(r<best+1e-9) nbest++; }
    }
  for(long c=0;c<N;c++) if(all[c]>=0){ double d=all[c]/(best-H); if(d<1.01) near[0]++; if(d<1.05) near[1]++; if(d<1.25) near[2]++; if(d<2) near[3]++; }
  printf("{\"p\":[%g,%g,%g],\"H\":%.9f,\"total\":%ld,\"best\":%.9f,\"nbest\":%ld,\"within1pct\":%ld,\"within5pct\":%ld,\"within25pct\":%ld,\"within2x\":%ld,\"arg\":[",p[0],p[1],p[2],H,total,best-H,nbest,near[0],near[1],near[2],near[3]);
  for(int i=0;i<L;i++) printf("%d%s",arg[i],i<L-1?",":"]}\n"); return 0; }
