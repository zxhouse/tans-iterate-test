// tANS encode/decode throughput with two tables of identical format; verifies round trip.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
typedef struct{uint16_t newX;uint8_t sym,nb;} D;
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); int L,n; long m; fread(&L,4,1,f); fread(&n,4,1,f); fread(&m,8,1,f);
  int *spread[2]; for(int t=0;t<2;t++){spread[t]=malloc(L*4); fread(spread[t],4,L,f);} int*msg=malloc(m*4); fread(msg,4,m,f); fclose(f);
  int R=0; while((1<<R)<L) R++;
  for(int t=0;t<2;t++){
    int*sym=spread[t]; int l[512]={0},occ[512]={0},start[512],k[512],*enc=malloc(L*4); D*dec=malloc(L*sizeof(D));
    for(int X=0;X<L;X++) l[sym[X]]++; int a=0; for(int s=0;s<n;s++){start[s]=a;a+=l[s]; int fl=0,v=l[s]; while(v>>1){v>>=1;fl++;} k[s]=R-fl;}
    for(int X=0;X<L;X++){int s=sym[X]; int y=l[s]+occ[s]; enc[start[s]+occ[s]]=X; occ[s]++; int fl=0,v=y; while(v>>1){v>>=1;fl++;} int nb=R-fl; dec[X].sym=s; dec[X].nb=nb; dec[X].newX=(y<<nb)-L;}
    uint64_t*buf=calloc(m*2+16,8); double best_e=1e9,best_d=1e9; long bits=0; int ok=1;
    for(int rep=0;rep<5;rep++){
      // encode backwards into a bit stack
      double t0=now(); long bp=0; uint32_t x=L;
      for(long i=m-1;i>=0;i--){int s=msg[i]; int nb=k[s]; if((x>>nb)<(uint32_t)l[s]) nb--; uint64_t v=x&((1u<<nb)-1);
        long w=bp>>6,o=bp&63; buf[w]|=v<<o; if(o+nb>64) buf[w+1]|=v>>(64-o); bp+=nb; x=enc[start[s]+(x>>nb)-l[s]]+L;}
      double te=now()-t0; if(te<best_e)best_e=te; bits=bp;
      // decode forwards popping the stack
      t0=now(); uint32_t X=x-L; long q=bp; long errs=0;
      for(long i=0;i<m;i++){D e=dec[X]; int nb=e.nb; q-=nb; long w=q>>6,o=q&63; uint64_t v=buf[w]>>o; if(o+nb>64) v|=buf[w+1]<<(64-o); v&=(1ull<<nb)-1;
        errs+= (e.sym!=msg[i]); X=e.newX+(uint32_t)v;}
      double td=now()-t0; if(td<best_d)best_d=td; if(errs||q!=0) ok=0;
      if(rep<4) memset(buf,0,(m*2+16)*8);
    }
    printf("{\"table\":%d,\"bits\":%ld,\"bps\":%.6f,\"enc_Msym_s\":%.1f,\"dec_Msym_s\":%.1f,\"roundtrip\":%d}\n",t,bits,(double)bits/m,m/best_e/1e6,m/best_d/1e6,ok);
  }
  return 0;}
