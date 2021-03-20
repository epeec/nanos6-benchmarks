#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <nanos6/debug.h>

#include "memory.h"

/* An OmpSs-2@Cluster implementation for the matrix-vector multiplication
 * kernel: y += A*x.
 * 
 * This version uses weak dependencies to decompose the work. It receives
 * as input the dimensions ([M, M]) of the problem, the task size (TS) and
 * number of rows per weak tasks (W). Finally, we can control the number of
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

void decompose_matvec(size_t M, double *A, size_t N, double *x, double *y,
		size_t TS)
{
	for (size_t i = 0; i < M; i += TS) {
		#pragma oss task in(A[i*N;TS*N]) in(x[0;N]) out(y[i;TS]) label("matvec")
		matvec(TS, &A[i * N], N, x, &y[i]);
	}
}

void decompose_init(size_t M, double *matrix, size_t N, size_t TS,
		size_t value)
{
	for (size_t i = 0; i < M; i += TS) {
		#pragma oss task out(matrix[i*N;TS*N]) label("init decomposed")
		init(N * TS, &matrix[i * N], value);
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
	fprintf(stderr, "usage: matvec_weak M N TS W ITER [CHECK]\n");
	return;
}

int main(int argc, char *argv[])
{
	size_t M, N, TS, W, ITER;
	double *A, *x, *y;
	bool check = false;
	
	struct timespec tp_start, tp_end;
	
	if (argc != 6 && argc != 7) {
		usage();
		return -1;
	}
	
	M = atoi(argv[1]);
	N = atoi(argv[2]);
	TS = atoi(argv[3]);
	W = atoi(argv[4]);
	ITER = atoi(argv[5]);
	
	/* The task size and the weak task size need to divide the
	 * number of rows*/
	if (M % TS) {
		fprintf(stderr, "The task-size needs to divide the number of"
				"rows\n");
		return -1;
	}
	
	if (M % W) {
		fprintf(stderr, "The weak task-size needs to divide the number "
				"of rows\n");
		return -1;
	}

	if (W % TS) {
		fprintf(stderr, "The task-size needs to divide the weak "
				"task-size\n");
	}
	
	if (argc == 7) {
		check = atoi(argv[6]);
	}
	
	A = (double*)nanos6_dmalloc(M * N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	x = (double*)nanos6_dmalloc(N * sizeof(double), nanos6_equpart_distribution, 0, NULL);
	y = (double*)nanos6_dmalloc(M * sizeof(double), nanos6_equpart_distribution, 0, NULL);

	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	#pragma oss task out(y[0;M]) label("initialize y")
	init(M, y, 0);
	
	#pragma oss task out(x[0;N]) label("initialize x")
	init(N, x, 1);
	
	for (size_t i = 0; i < M; i += W) {
		#pragma oss task weakout(A[i*N;W*N]) label("initialize A")
		decompose_init(W, &A[i * N], N, TS, 2);
	}
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		for (size_t i = 0; i < M; i += W) {
			#pragma oss task weakin(A[i*N;W*N]) weakin(x[0;N]) weakinout(y[i;W]) label("decompose matvec")
			decompose_matvec(W, &A[i * N], N, x, &y[i], TS);
		}
	}
	
	if (check) {
		#pragma oss task in(A[0;M*N]) in(x[0;N]) in(y[0;M]) label("check_result")
		check_result(M, A, N, x, y, ITER);
	}
	
	#pragma oss taskwait

	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ ((double)(tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
	
	double mflops =
		ITER *			/* ITER times of kernel FLOPS */
		3 * M * N 		/* 3 operations for every element of A */
		/ (time_msec / 1000.0) 	/* time in seconds */
		/ 1e6; 			/* convert to Mega */

	printf("M:%zu N:%zu TS:%zu WEAK_TS: %zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
		M, N, TS, W, ITER, nanos6_get_num_cluster_nodes(), nanos6_get_num_cpus(),
		time_msec, mflops);
	
	nanos6_dfree(A, M * N * sizeof(double));
	nanos6_dfree(x, N * sizeof(double));
	nanos6_dfree(y, M * sizeof(double));

	return 0;
}
