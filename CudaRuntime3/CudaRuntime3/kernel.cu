#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define ll long long
#define MAX_TREAD_COUNT 1024
#define SHARED_MEMORY_SIZE MAX_TREAD_COUNT * sizeof(ll)

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

__host__ void fillArray(ll* arr, const size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		arr[i] = i + 1;
	}
}

__host__ void printArray(ll* arr, const size_t size)
{
	printf("[ ");
	for (size_t i = 0; i < size; ++i)
	{
		printf("%lld ", arr[i]);
	}
	printf("]\n");
}

__host__ void printDeviceProperties(const cudaDeviceProp& deviceProp)
{
	// Основная информация
	printf("\n\nGPU: %s\n", deviceProp.name);
	printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / 1073741824.0);

	// Блоки и сетка
	printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Max Block Dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max Grid Dim: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	// Аппаратные характеристики
	printf("Max blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
	printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);
	printf("Clock Rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);
	printf("Shared Memory per Block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);

	// Дополнительно
	printf("Warp Size: %d\n", deviceProp.warpSize);
	printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
	printf("Integrated GPU: %s\n\n\n", deviceProp.integrated ? "Yes" : "No");
}

// Фаза Up-sweep (редукция)
__global__ void upsweep_kernel(ll* arr, int size, int stride)
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	// Должны работать потоки с индексами: stride*2-1, stride*4-1, stride*6-1, ...
	if (threadId >= size)
	{
		return;
	}

	// Правильное условие: поток должен быть на позиции (k * 2 * stride - 1)
	if ((threadId + 1) % (2 * stride) == 0)
	{
		int left_idx = threadId - stride;
		if (left_idx >= 0)
		{
			arr[threadId] += arr[left_idx];
		}
	}
}

// Фаза Down-sweep (распространение)
__global__ void downsweep_kernel(ll* arr, size_t size, int stride)
{
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId >= size)
	{
		return;
	}

	// Потоки должны быть на позициях: stride*2-1, stride*4-1, stride*6-1, ...
	if ((threadId + 1) % (2 * stride) == 0)
	{
		int left_idx = threadId - stride;
		if (left_idx >= 0) 
		{
			// Сохраняем значение левого элемента
			ll temp = arr[left_idx];
			// Перемещаем текущее значение в левый элемент
			arr[left_idx] = arr[threadId];
			// Добавляем сохраненное значение к текущему
			arr[threadId] += temp;
		}
	}
}

__host__ cudaError_t prefixAmount(ll* source, int size)
{
	cudaError_t cudaStatus;
	for (int stride = 1; stride < size; stride *= 2)
	{
		int blocks_per_grid = ceill((double)size / MAX_TREAD_COUNT);
		if (blocks_per_grid == 0)
		{
			blocks_per_grid = 1;
		}

		upsweep_kernel<<<blocks_per_grid, MAX_TREAD_COUNT>>>(source, size, stride);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			return cudaStatus;
		}
	}

	ll total_sum;
	// Сохраняем общую сумму и обнуляем последний элемент
	cudaMemcpy(&total_sum, &source[size - 1], sizeof(ll), cudaMemcpyDeviceToHost);
	cudaMemset(&source[size - 1], 0, sizeof(ll));

	for (int stride = size / 2; stride >= 1; stride /= 2)
	{
		int blocks_per_grid = ceill((double)size / MAX_TREAD_COUNT);
		if (blocks_per_grid == 0)
		{
			blocks_per_grid = 1;
		}

		downsweep_kernel << <blocks_per_grid, MAX_TREAD_COUNT >> > (source, size, stride);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			return cudaStatus;
		}
	}
	return cudaSuccess;
}

int main()
{
	srand(time(NULL));

	ll* source = NULL, * devSource = NULL, * result = NULL, * devResult = NULL;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaDeviceProp deviceProp;
	float milliseconds = 0;

	const size_t arraySize = 8;
	const dim3 blockDim(MAX_TREAD_COUNT), gridDim((size_t)ceil(arraySize / ((double)blockDim.x)));

	source = new ll[arraySize];
	fillArray(source, arraySize);
	printArray(source, arraySize);

	///////////////////////////////////////GPU/////////////////////////////////////////////////////
	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetDeviceProperties failed!");

	printDeviceProperties(deviceProp);

	cudaStatus = cudaMalloc(&devSource, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devSource failed!");

	cudaStatus = cudaMalloc(&devResult, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMemcpy(devSource, source, arraySize * sizeof(ll), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devSource failed!");

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	cudaStatus = prefixAmount(devSource, arraySize);
	CHECK_CUDA_ERROR(cudaStatus, "prefixAmount failed!");

	cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetLastError failed!");

	cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR(cudaStatus, "cudaDeviceSynchronize failed!");

	cudaStatus = cudaEventRecord(stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&stop) failed!");

	// Ждем завершения всех операций
	cudaStatus = cudaEventSynchronize(stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventSynchronize(&stop) failed!");

	cudaStatus = cudaEventElapsedTime(&milliseconds, start, stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventElapsedTime failed!");

	result = new ll[arraySize];
	cudaStatus = cudaMemcpy(result, devSource, arraySize * sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(&result failed!");

	printArray(result, arraySize);
	printf("GPU time: %f ms\n", milliseconds);

Finish:
	DELETE_ARRAY_IF_EXISTS(source);
	DELETE_ARRAY_IF_EXISTS(result);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	if (devSource)
	{
		cudaStatus = cudaFree(devSource);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devSource failed!");
	}

	if (devResult)
	{
		cudaStatus = cudaFree(devResult);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devResult failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");

	system("pause");
	return 0;
}
#endif
