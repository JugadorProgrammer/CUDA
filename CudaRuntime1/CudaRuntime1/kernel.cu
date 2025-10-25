#pragma once
#ifndef __longELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>

// Функция для чтения массива long из бинарного файла
void fillArray(long long* arr, size_t length)
{
	for (size_t i = 0; i < length; ++i)
	{
		arr[i] = i;
	}
}

__global__ void reduction(long long* arr, long long* result, size_t size)
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

__global__ void reductionWithShared(long long* arr, long long* result, size_t size)
{
	extern __shared__ long long sharedArr[];
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

long main()
{
	srand(time(NULL));
	const int iterationCount = 100;
	double averageTime = 0.0;
	for (size_t i = 0; i < iterationCount; ++i)
	{
		const long threadsCount = 1024;
		const long sharedMemorySize = threadsCount * 2 * sizeof(long long);
		const size_t arraySize = 1000000;
		const size_t byteSize = arraySize * sizeof(long long);
		const size_t threadsCountBytes = threadsCount * sizeof(long long);
		long long* arr = new long long[arraySize];
		long long* result = new long long();
		long long* devArr = 0;
		long long* devResult = 0;
		cudaError_t cudaStatus;

		fillArray(arr, arraySize);
		// Создаем события для измерения времени
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaDeviceProp deviceProp;
		cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceProperties failed!");
			goto Finish;
		}

		// Инициализируем переменные
		cudaStatus = cudaMalloc(&devArr, byteSize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Finish;
		}

		cudaStatus = cudaMalloc(&devResult, sizeof(long long));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Finish;
		}

		cudaStatus = cudaMemcpy(devArr, arr, byteSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Finish;
		}

		// Записываем начальное время
		cudaEventRecord(start);	

		//reductionWithShared<<<1, threadsCount, sharedMemorySize>>>(devArr, devResult, arraySize);
		reduction<<<1, threadsCount>>>(devArr, devResult, arraySize);

		// Записываем конечное время
		cudaEventRecord(stop);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Finish;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Finish;
		}

		// Вычисляем время выполнения
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaStatus = cudaMemcpy(result, devResult, sizeof(long long), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Finish;
		}

		printf("Result = %lld Time = %f ms\n", *result, milliseconds);
		averageTime += milliseconds;
	Finish:
		delete[] arr;
		// Освобождаем ресурсы
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		// Add vectors in parallel.
		//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}

	printf("Average Time = %f ms\n", averageTime / iterationCount);

	system("pause");
	return 0;
}
#endif