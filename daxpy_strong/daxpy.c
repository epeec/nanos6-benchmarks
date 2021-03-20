#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <nanos6/debug.h>

#include "common.h"

/* An OmpSs-2@Cluster implementation of the daxpy BLAS operation
 * 
 * This version uses strong dependencies. It receives as input the
 * dimension N of the the vectors and the task size (TS). It divides
 * the problem in N / TS (strong) tasks. Finally, we can control the
 * number of times that the daxpy kernel will execute with the ITER
 * argument.
 * 
 * We initialize the vectors with prefixed values which we can later
 * check to ensure correctness of the computations
 */

void daxpy(size_t N, double *x, double alpha, double *y)
{
	for (size_t i = 0; i < N; ++i) {
		y[i] += alpha * x[i];
	}
}

void init(size_t N, double *vector, double value)
{
	for (size_t i = 0; i < N; ++i) {
		vector[i] = value;
	}
}

void check_result(size_t N, double *x, double alpha, double *y, size_t ITER)
{
	double *y_serial = (double*)nanos6_lmalloc(N * sizeof(double));
	init(N, y_serial, 0);
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		daxpy(N, x, alpha, y_serial);
	}
	
	for (size_t i = 0; i < N; ++i) {
		if (y_serial[i] != y[i]) {
			printf("FAILED\n");
			nanos6_lfree(y_serial, N * sizeof(double));
			return;
		}
	}
	
	printf("SUCCESS\n");
	nanos6_lfree(y_serial, N * sizeof(double));
}

void usage()
{
	fprintf(stderr, "usage: daxpy_strong N TS ITER [CHECK]\n");
	return;
}

int main(int argc, char *argv[])
{
	size_t N, TS, ITER;
	double alpha, *x, *y;
	bool check = false;
	struct timespec tp_start, tp_end;
	
	if (argc != 4 && argc != 5) {
		usage();
		return -1;
	}
	
	N = atoi(argv[1]);
	TS = atoi(argv[2]);
	ITER = atoi(argv[3]);
	
	/* The task size needs to divide the number of
	 * elements */
	if (N % TS) {
		fprintf(stderr, "The task-size needs to divide the vector size\n");
		return -1;
	}
	
	if (argc == 5) {
		check = atoi(argv[4]);
	}
	
	x = (double*)nanos6_dmalloc(N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	y = (double*)nanos6_dmalloc(N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	#pragma oss task out(y[0;N]) label("initialize y")
	init(N, y, 0);
	
	#pragma oss task out(x[0;N]) label("initialize x")
	init(N, x, 42);
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		for (size_t i = 0; i < N; i += TS) {
			#pragma oss task in(x[i;TS]) inout(y[i;TS]) \
				label("daxpy task")
			daxpy(TS, x + i, alpha, y + i); 
		}
	}
	
	#pragma oss taskwait
	
	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	if (check) {
		#pragma oss task in(x[0;N]) inout(y[0;N]) label("check result")
		check_result(N, x, alpha, y, ITER);
	
		#pragma oss taskwait
	}

	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ ((tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
	
	double mflops =
		ITER 			/* 'ITER' times of kernel FLOPS */
		* 3 * N 		/* 3 operations for every vector element */
		/ (time_msec / 1000.0) 	/* time in seconds */
		/ 1e6;			/* convert to Mega */
	
	printf("N:%zu TS:%zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
		N, TS, ITER, nanos6_get_num_cluster_nodes(), nanos6_get_num_cpus(),
		time_msec, mflops);
	
	nanos6_dfree(x, N * sizeof(double));
	nanos6_dfree(y, N * sizeof(double));
	
	return 0;

	return 0;
}
