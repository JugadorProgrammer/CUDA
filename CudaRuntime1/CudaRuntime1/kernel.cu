#pragma once
#ifndef __longELLISENSE_
#define NOMINMAX

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define ll long long
#define minimum(a, b) (((a) < (b)) ? (a) : (b))
#define TILE_SIZE (size_t)MAX_2D_TREAD_COUNT
#define CALC_TIME_MS(start, end) (((double)((end) - (start)) * 1000.0) / CLOCKS_PER_SEC)
#define MAX_GPU_THREADS_PER_BLOCK 1024.0
#define MAX_CPU_LOGICAL_THREAD 8

#define DELETE_IF_EXISTS(ptr) \
    if (ptr) \
    { \
        delete ptr; \
    }

#define DELETE_ARRAY_IF_EXISTS(ptr) \
    if (ptr) \
    { \
        delete[] ptr; \
    }

#define CHECK_CUDA_ERROR(cudaStatus, message) \
    if ((cudaStatus) != cudaSuccess) \
    { \
        fprintf(stderr, message); \
		fprintf(stderr, "CUDA error string:  %s\n", cudaGetErrorString(cudaStatus)); \
        goto Finish; \
    }

#define PRINT_CUDA_ERROR(cudaStatus, message) \
    if ((cudaStatus) != cudaSuccess) \
    { \
        fprintf(stderr, message); \
		fprintf(stderr, "CUDA error string:  %s\n", cudaGetErrorString(cudaStatus)); \
    }

typedef struct
{
	ll* array;
	ll start;
	ll end;
	ll result;
} ThreadData;

// Функция для чтения массива long из бинарного файла
void fillArray(ll* arr, size_t length)
{
	for (size_t i = 0; i < length; ++i)
	{
		arr[i] = i;
	}
}

void printArray(ll* arr, size_t length)
{
	printf("[ ");
	for (size_t i = 0; i < length; ++i)
	{
		printf("%d ", arr[i]);
	}
	printf(" ]\n");
}

__host__  DWORD WINAPI worker(LPVOID param)
{
	ThreadData* data = (ThreadData*)param;
	data->result = 0;

	for (int i = data->start; i < data->end; ++i)
	{
		data->result += data->array[i];
	}

	return 0;
}

__host__ ll parallel_reduction(ll* array, size_t size, int num_threads)
{
	if (num_threads <= 0 || num_threads > size)
	{
		num_threads = size;
	}

	HANDLE* threads = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
	ThreadData* data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));

	int chunk = ceil((double)size / num_threads);
	for (int i = 0; i < num_threads; ++i)
	{
		data[i].array = array;
		data[i].start = i * chunk;
		data[i].end = (i == num_threads - 1) ? size : data[i].start + chunk;

		threads[i] = CreateThread(NULL, 0, worker, &data[i], 0, NULL);
	}

	WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

	int total = 0;
	for (int i = 0; i < num_threads; i++)
	{
		total += data[i].result;
		CloseHandle(threads[i]);
	}

	free(threads);
	free(data);
	return total;
}

__global__ void reduction(ll* arr, ll* result, size_t size)
{
	unsigned long threadCount = blockDim.x;
	unsigned long threadIdxX = threadIdx.x;
	// Сделаем так, чтобы кол-во потоков равнялось кол-ву элементов в массиве
	long dif = ceil((double)(size) / (double)(threadCount));
	if (dif > 2)
	{
		long step = ceil((double)(size - threadCount * 2) / (double)(threadCount));
		for (size_t i = threadCount * 2 + step * threadIdxX, j = 0; j < step; ++i, ++j)
		{
			arr[threadIdxX] += arr[i];
		}

		size = threadCount * 2;
	}

	__syncthreads();
	while (size > 1)
	{
		if (threadIdxX >= size / 2)
		{
			return;
		}

		if (threadIdxX == 0 && size % 2 != 0)
		{
			arr[0] += arr[--size];
		}

		arr[threadIdxX] += arr[size - 1 - threadIdxX];
		__syncthreads();
		size /= 2;
	}

	if (threadIdxX == 0)
	{
		(*result) = arr[0];
	}
}

__global__ void reductionWithShared(ll* arr, ll* result, size_t size)
{
	extern __shared__ ll sharedArr[];
	unsigned long threadCount = blockDim.x;
	unsigned long threadIdxX = threadIdx.x;

	sharedArr[threadIdxX] = arr[threadIdxX];
	// Сделаем так, чтобы кол-во потоков равнялось кол-ву элементов в массиве
	long dif = ceil((double)(size) / (double)(threadCount));
	if (dif > 2)
	{
		long step = ceil((double)(size - threadCount * 2) / (double)(threadCount));
		for (size_t i = threadCount * 2 + step * threadIdxX, j = 0; j < step; ++i, ++j)
		{
			sharedArr[threadIdxX] += arr[i];
		}

		size = threadCount * 2;
	}

	if (dif > 1 && size - 1 - threadIdxX > ceil(((double)threadCount) / 2.0))
	{
		sharedArr[size - 1 - threadIdxX] = arr[size - 1 - threadIdxX];
	}

	__syncthreads();
	while (size > 1)
	{
		if (threadIdxX >= size / 2)
		{
			return;
		}

		if (threadIdxX == 0 && size % 2 != 0)
		{
			sharedArr[0] += sharedArr[--size];
		}

		sharedArr[threadIdxX] += sharedArr[size - 1 - threadIdxX];
		__syncthreads();
		size /= 2;
	}

	if (threadIdxX == 0)
	{
		(*result) = sharedArr[0];
	}
}

__host__ void CPU(ll* arr, const size_t arraySize)
{
	clock_t start, end;
	double cpu_time_used;

	start = clock();
	
	ll result = parallel_reduction(arr, arraySize, MAX_CPU_LOGICAL_THREAD);

	end = clock();
	float milliseconds = CALC_TIME_MS(start, end);
	printf("CPU: Result = %lld Time = %f ms\n", result, milliseconds);
}

__host__ void GPU(ll* arr, const size_t arraySize)
{
	ll* result = NULL, * devResult = NULL, * devArr = NULL;
	size_t byteSize = arraySize * sizeof(ll);

	cudaError_t cudaStatus;
	// Создаем события для измерения времени
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetDeviceProperties failed!");

	// Инициализируем переменные
	cudaStatus = cudaMalloc(&devArr, byteSize);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc(&devResult, sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(devArr, arr, byteSize, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	// Записываем начальное время
	cudaEventRecord(start);

	int threadsCount = minimum(MAX_GPU_THREADS_PER_BLOCK, arraySize);
	reduction<<<1, threadsCount>>>(devArr, devResult, arraySize);

	// Записываем конечное время
	cudaEventRecord(stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR(cudaStatus, "addKernel launch failed\n");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaStatus, "cudaDeviceSynchronize returned error after launching addKernel!");

	// Вычисляем время выполнения
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	result = new ll();
	cudaStatus = cudaMemcpy(result, devResult, sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	printf("GPU: Result = %lld Time = %f ms\n", *result, milliseconds);
Finish:
	DELETE_IF_EXISTS(result);
	// Освобождаем ресурсы
	cudaEventDestroy(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaEventDestroy(stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	CHECK_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ void GPUShared(ll* arr, const size_t arraySize)
{
	ll* result = NULL, * devResult = NULL, * devArr = NULL;
	size_t byteSize = arraySize * sizeof(ll);

	cudaError_t cudaStatus;
	// Создаем события для измерения времени
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetDeviceProperties failed!");

	// Инициализируем переменные
	cudaStatus = cudaMalloc(&devArr, byteSize);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc(&devResult, sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(devArr, arr, byteSize, cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	// Записываем начальное время
	cudaEventRecord(start);

	int threadsCount = minimum(MAX_GPU_THREADS_PER_BLOCK, arraySize);
	const int sharedMemorySize = threadsCount * 2 * sizeof(ll);
	reductionWithShared<<<1, threadsCount, sharedMemorySize>>>(devArr, devResult, arraySize);

	// Записываем конечное время
	cudaEventRecord(stop);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR(cudaStatus, "addKernel launch failed\n");

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaStatus, "cudaDeviceSynchronize returned error after launching addKernel!");

	// Вычисляем время выполнения
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	result = new ll();
	cudaStatus = cudaMemcpy(result, devResult, sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	printf("GPU_SHARED: Result = %lld Time = %f ms\n", *result, milliseconds);
Finish:
	DELETE_IF_EXISTS(result);
	// Освобождаем ресурсы
	cudaEventDestroy(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaEventDestroy(stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	CHECK_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

long main()
{
	srand(time(NULL));
	const size_t arraySize = 9000;
	ll* arr = new ll[arraySize];

	fillArray(arr, arraySize);
	//printArray(arr, arraySize);
	
	CPU(arr, arraySize);
	GPU(arr, arraySize);
	GPUShared(arr, arraySize);

	delete[] arr;
	system("pause");
	return 0;
}
#endif