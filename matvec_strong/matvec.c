#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <nanos6/debug.h>

#include "common.h"

/* An OmpSs-2@Cluster implementation for the matrix-vector multiplication
 * kernel: y += A*x.
 * 
 * This version does not make use of weak dependencies. It receives as input
 * the dimensions ([M, N]) of the problem and the task size (TS). It divides
 * the problem in M / TS (strong) tasks. Finally, we can control the number of
 * times that we will execute the matvec kernel with the ITER argument.
 * 
 * We initialize the vectors with prefixed values which we can later check to
 * ensure the correctness of the computation.
 */


void matvec(size_t M, double *A, size_t N, double *x, double *y)
{
	for (size_t i = 0; i < M; ++i) {
		double res = 0.0;
		for (size_t j = 0; j < N; ++j) {
			res += A[i * N + j] * x[j];
		}
		
		y[i] += res;
	}
}

void init(size_t M, double *vec, double value)
{
	for (size_t i = 0; i < M; ++i) {
		vec[i] = value;
	}
}

void check_result(size_t M, double *A, size_t N, double *x, double *y,
		size_t ITER)
{
	double *y_serial = (double*)nanos6_lmalloc(M * sizeof(double));
	init(M, y_serial, 0);
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		matvec(M, A, N, x, y_serial);
	}
	
	for (size_t i = 0; i < M; ++i) {
		if (y_serial[i] != y[i]) {
			printf("FAILED\n");
			nanos6_lfree(y_serial, M * sizeof(double));
			return;
		}
	}
	
	printf("SUCCESS\n");
	nanos6_lfree(y_serial, M * sizeof(double));
}

void usage()
{
	fprintf(stderr, "usage: matvec_strong M N TS ITER [CHECK]\n");
	return;
}

int main(int argc, char *argv[])
{
	size_t M, N, TS, ITER;
	double *A, *x, *y;
	int check = false;
	struct timespec tp_start, tp_end;

	if (argc != 5 && argc != 6) {
		usage();
		return -1;
	}
	
	M = atoi(argv[1]);
	N = atoi(argv[2]);
	TS = atoi(argv[3]);
	ITER = atoi(argv[4]);
	
	/* The task size needs to divide the number
	 * of rows */
	if (M % TS) {
		fprintf(stderr, "The task-size needs to divide the number of rows\n");
		return -1;
	}
	
	if (argc == 6) {
		check = atoi(argv[5]);
	}
	
	A = (double*)nanos6_dmalloc(M * N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	x = (double*)nanos6_dmalloc(N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	y = (double*)nanos6_dmalloc(M * sizeof(double), nanos6_equpart_distribution, 0, NULL);

	clock_gettime(CLOCK_MONOTONIC, &tp_start);

	#pragma oss task out(y[0;M]) label("initialize y")
	init(M, y, 0);
	
	#pragma oss task out(x[0;N]) label("initialize x")
	init(N, x, 1);
	
	for (size_t i = 0; i < M; i += TS) {
		#pragma oss task out(A[i*N;N*TS]) label("initialize A")
		init(N * TS, &A[i * N], 2);
	}
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		for (size_t i = 0; i < M; i += TS) {
			#pragma oss task in(A[i*N;N*TS]) in(x[0;N]) inout(y[i;TS]) label("matvec task")
			matvec(TS, &A[i*N], N, x, &y[i]);
		}
	}
	
	if (check) {
		#pragma oss task in(A[0;M*N]) in(x[0;N]) in(y[0;M]) label("check result")
		check_result(M, A, N, x, y, ITER);
	}
	
	#pragma oss taskwait

	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ ((double)(tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
	
	double mflops =
		ITER 			/* 'ITER' times of kernel FLOPS */
		* 3 * M * N 		/* 3 operations for every element of A */
		/ (time_msec / 1000.0) 	/* time in seconds */
		/ 1e6; 			/* convert to Mega */

	printf("M:%zu N:%zu TS:%zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
		M, N, TS, ITER, nanos6_get_num_cluster_nodes(), nanos6_get_num_cpus(),
		time_msec, mflops);
	
	nanos6_dfree(A, M * N * sizeof(double));
	nanos6_dfree(x, N * sizeof(double));
	nanos6_dfree(y, M * sizeof(double));
	
	return 0;
}
