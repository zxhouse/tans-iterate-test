// tANS exact-rate evaluation, sampled evaluation and certified local search.
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAXL 8192
#define MAXN 512
typedef struct { int lo[MAXL], hi[MAXL], nb[MAXL]; double ps[MAXL]; } Tab;
static int ilog2(int v){int f=0; while(v>>1){v>>=1;f++;} return f;}
// build per-position transition description. returns 0 if invalid (a symbol with zero slots)
static int build(const int*sym,int L,int n,const double*p,Tab*t){
  int l[MAXN]={0}, occ[MAXN]={0}; int R=ilog2(L);
  for(int X=0;X<L;X++) l[sym[X]]++;
  for(int s=0;s<n;s++) if(l[s]==0) return 0;
  for(int X=0;X<L;X++){ int s=sym[X]; int y=l[s]+occ[s]++; int k=R-ilog2(l[s]); int nb=k;
    if(((long)y<<k)>=2L*L) nb=k-1;
    t->lo[X]=(y<<nb)-L; t->hi[X]=((y+1)<<nb)-L; t->nb[X]=nb; t->ps[X]=p[s]; }
  return 1;
}
// exact expected bits/symbol for an i.i.d. source p; pi is warm start in, stationary out
double TOL=1e-14;
double exact_rate(const int*sym,int L,int n,const double*p,double*pi,int warm){
  static Tab t; static double P[MAXL+1], q[MAXL];
  if(!build(sym,L,n,p,&t)) return 1e9;
  if(!warm) for(int X=0;X<L;X++) pi[X]=1.0/L;
  for(int it=0;it<20000;it++){
    P[0]=0; for(int X=0;X<L;X++) P[X+1]=P[X]+pi[X];
    double d=0;
    for(int X=0;X<L;X++){ q[X]=t.ps[X]*(P[t.hi[X]]-P[t.lo[X]]); d+=fabs(q[X]-pi[X]); }
    for(int X=0;X<L;X++) pi[X]=0.5*(pi[X]+q[X]);
    if(d<TOL) break;
  }
  double sum=0; for(int X=0;X<L;X++) sum+=pi[X];
  for(int X=0;X<L;X++) pi[X]/=sum;
  P[0]=0; for(int X=0;X<L;X++) P[X+1]=P[X]+pi[X];
  double r=0; for(int X=0;X<L;X++) r+=t.nb[X]*t.ps[X]*(P[t.hi[X]]-P[t.lo[X]]);
  return r;
}
// encode m symbols of a trace starting at *state (X coordinate); returns total bits
double trace_bits(const int*sym,int L,int n,const int*trace,long m,int*state){
  static int enc[MAXL]; int l[MAXN]={0}, start[MAXN], occ[MAXN]={0}, k[MAXN]; int R=ilog2(L);
  for(int X=0;X<L;X++) l[sym[X]]++;
  int a=0; for(int s=0;s<n;s++){ start[s]=a; a+=l[s]; k[s]= l[s]? R-ilog2(l[s]) : 0; }
  for(int X=0;X<L;X++){ int s=sym[X]; enc[start[s]+occ[s]++]=X; }
  long bits=0; int x=*state+L;
  for(long i=0;i<m;i++){ int s=trace[i]; int nb=k[s]; if((x>>nb)<l[s]) nb--; bits+=nb; x=enc[start[s]+(x>>nb)-l[s]]+L; }
  *state=x-L; return (double)bits;
}
// certified local search: sweeps over swap (|i-j|<=w) and optional local reassign moves
// until a full sweep yields no improvement. Objective: sum_c wts[c]*rate(p_c). returns final objective.
double local_search(int*sym,int L,int n,const double*p,int ncls,const double*wts,int w,int reassign,int maxsweeps,long*evals,int*sweeps){
  static double pi[4][MAXL], tmp[4][MAXL]; double best=0; long ev=0;
  for(int c=0;c<ncls;c++) best+=wts[c]*exact_rate(sym,L,n,p+c*n,pi[c],0);
  int l[MAXN]={0}; for(int X=0;X<L;X++) l[sym[X]]++;
  int sw=0;
  for(;sw<maxsweeps;sw++){ int improved=0;
    for(int i=0;i<L;i++){
      for(int d=1;d<=w && i+d<L;d++){ int j=i+d; if(sym[i]==sym[j]) continue;
        int a=sym[i]; sym[i]=sym[j]; sym[j]=a; double r=0;
        for(int c=0;c<ncls;c++){ memcpy(tmp[c],pi[c],L*sizeof(double)); r+=wts[c]*exact_rate(sym,L,n,p+c*n,tmp[c],1);} ev++;
        if(r<best-1e-11){ best=r; improved=1; for(int c=0;c<ncls;c++) memcpy(pi[c],tmp[c],L*sizeof(double)); }
        else { a=sym[i]; sym[i]=sym[j]; sym[j]=a; }
      }
      if(reassign){ int a0=sym[i];
        for(int j=(i-w<0?0:i-w); j<=i+w && j<L; j++){ int b=sym[j]; int a=sym[i]; if(b==a||l[a]<=1) continue;
          int seen=0; for(int q=(i-w<0?0:i-w);q<j;q++) if(sym[q]==b){seen=1;break;} if(seen) continue;
          sym[i]=b; double r=0;
          for(int c=0;c<ncls;c++){ memcpy(tmp[c],pi[c],L*sizeof(double)); r+=wts[c]*exact_rate(sym,L,n,p+c*n,tmp[c],1);} ev++;
          if(r<best-1e-11){ best=r; improved=1; l[a]--; l[b]++; for(int c=0;c<ncls;c++) memcpy(pi[c],tmp[c],L*sizeof(double)); }
          else sym[i]=a;
        } (void)a0;
      }
    }
    if(!improved){ sw++; break; }
  }
  *evals=ev; *sweeps=sw; return best;
}
