#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gauss_2d_sphere.h"

#ifndef FABS
	#define FABS(a) ((a)>=0?(a):-(a))
#endif


double f(double x, double y, void* data)
{
	return sin(x*x+y*y);
}

double f1(double x, double y, void* data)
{
  //printf("x=%.15g, y=%.15g\n",x,y);
  return x*x;
}

int main(int argc, char* argv[])
{

	/* numerical approximation of integral */
	double approx;		

	/* true value of int(sin(x^2+y^2), x^2+y^2<=1) = Pi-cos(1)*Pi*/
	//double exact = 1.4441828987568200687715608702750; 

	/* approximation error */
	//double error;       

	int i;
/*
	printf("Numerical Approximation of int(sin(x^2+y^2), x^2+y^2<=1) by Gauss product rule:\n");
	for (i=1;i<=50;i++)
	{
		approx = gauss_product_2D_sphere(i,f,NULL,1.0,0.0,0.0);
		error = approx-exact;
		printf("n = %4d: error = %.15g\n",i,FABS(error));
	}

	for (i=1;i<=16;i++)
	{
		approx = gauss_product_2D_sphere(64*i,f,NULL,1.0,0.0,0.0);
		error = approx-exact;
		printf("n = %4d: error = %.15g\n",64*i,FABS(error));
	}
*/
        printf("Approximate integral of x*x by Gauss product rule:\n");
        for (i=3;i<=3;i++)
        {
          approx = gauss_product_2D_sphere(i,f1,NULL,1.0,0.0,0.0);
          printf("n = %4d: value = %.15g\n",i,approx);
        }

}
