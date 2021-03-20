#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <nanos6/debug.h>

#pragma oss task out(fn[0]) label("fibo")
void fibonacci(size_t n, size_t *fn)
{
	if (n <= 1) {
		*fn = n;
		return;
	}
	
	size_t fn1, fn2;
	
	fibonacci(n - 1, &fn1);
	
	fibonacci(n - 2, &fn2);
	
	#pragma oss taskwait
	*fn = fn1 + fn2;
}

#pragma oss task in(fn[0])
void check_result(size_t n, size_t *fn)
{
	size_t fn_serial;
	if (n <= 1) {
		fn_serial = n;
	} else {
		size_t fn2 = 0;
		size_t fn1 = 1;
		for (size_t i = 1; i < n; ++i) {
			fn_serial = fn1 + fn2;
			fn2 = fn1;
			fn1 = fn_serial;
		}
	}
	
	if (fn_serial == *fn) {
		printf("SUCCESS\n");
	} else {
		printf("FAILED (EXPECTED:%lu GOT:%lu)\n", fn_serial, *fn);
	}
}

void usage()
{
	fprintf(stderr, "usage: fibonacci N [CHECK]\n");
}

int main(int argc, char *argv[])
{
	size_t n, fn;
	bool check = false;
	struct timespec tp_start, tp_end;
	
	if (argc != 2 && argc != 3) {
		usage();
		return -1;
	}
	
	n = atoi(argv[1]);
	
	if (argc == 3) {
		check = atoi(argv[2]);
	}
	
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	fibonacci(n, &fn);
	
	#pragma oss taskwait
	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	if (check) {
		check_result(n, &fn);
		#pragma oss taskwait
	}
	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ (tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6;
	
	printf("N:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf\n",
			n, nanos6_get_num_cluster_nodes(),
			nanos6_get_num_cpus(),
			time_msec);
	
	return 0;
}
