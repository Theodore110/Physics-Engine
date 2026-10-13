#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern float k_run(int,int);
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  int n=argc>1?atoi(argv[1]):200, steps=argc>2?atoi(argv[2]):60, reps=argc>3?atoi(argv[3]):7;
  double t[64]; float cs=0; k_run(n,3); /* warmup */
  for(int r=0;r<reps;r++){double a=now();cs=k_run(n,steps);t[r]=now()-a;}
  qsort(t,reps,sizeof(double),cmp);
  printf("min %.1f ms  median %.1f ms  checksum %.6f\n",t[0]*1e3,t[reps/2]*1e3,cs);
}
