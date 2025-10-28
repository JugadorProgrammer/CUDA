#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string.h>

#define ll long long
#define MAX_TREAD_COUNT 1024
#define CALC_TIME_MS(start, end) (((double)((end) - (start)) * 1000.0) / CLOCKS_PER_SEC)
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
void upsweep_cpu(ll* arr, int size, int stride)
{
	for (int threadId = 0; threadId < size; threadId++)
	{
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
}

// Фаза Down-sweep (распространение)
void downsweep_cpu(ll* arr, size_t size, int stride)
{
	for (int threadId = 0; threadId < size; threadId++)
	{
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
}

__host__ void prefixAmount_cpu(ll** arr, int size)
{
	ll* source = *arr;
	// Фаза Up-sweep
	for (int stride = 1; stride < size; stride *= 2)
	{
		upsweep_cpu(source, size, stride);
	}

	ll total_sum;
	// Сохраняем общую сумму и обнуляем последний элемент
	total_sum = source[size - 1];
	source[size - 1] = 0;

	// Фаза Down-sweep
	for (int stride = size / 2; stride >= 1; stride /= 2)
	{
		downsweep_cpu(source, size, stride);
	}

	ll* result = new ll[size];

	memcpy(result, source + 1, (size - 1) * sizeof(ll));
	memcpy(result + (size - 1), &total_sum, sizeof(ll));
	*arr = result;
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

__host__ cudaError_t prefixAmount(ll** arr, int size)
{
	cudaError_t cudaStatus;
	ll* source = *arr;
	for (int stride = 1; stride < size; stride *= 2)
	{
		int num_threads_needed = (size + (2 * stride) - 1) / (2 * stride);
		int blocks_per_grid = (num_threads_needed + MAX_TREAD_COUNT - 1) / MAX_TREAD_COUNT;
		if (blocks_per_grid == 0)
		{
			blocks_per_grid = 1;
		}

		upsweep_kernel << <blocks_per_grid, MAX_TREAD_COUNT >> > (source, size, stride);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			return cudaStatus;
		}
	}

	ll total_sum;
	// Сохраняем общую сумму и обнуляем последний элемент
	cudaStatus = cudaMemcpy(&total_sum, &source[size - 1], sizeof(ll), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	cudaStatus = cudaMemset(&source[size - 1], 0, sizeof(ll));
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	for (int stride = size / 2; stride >= 1; stride /= 2)
	{
		int num_threads_needed = (size + (2 * stride) - 1) / (2 * stride);
		int blocks_per_grid = (num_threads_needed + MAX_TREAD_COUNT - 1) / MAX_TREAD_COUNT;

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

	ll* result;
	cudaStatus = cudaMalloc(&result, size * sizeof(ll));
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(result, source + 1, (size - 1) * sizeof(ll), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	cudaStatus = cudaMemset(result + (size - 1), total_sum, 1);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	*arr = result;
	return cudaSuccess;
}

__host__ void CPU(ll* arr, const size_t arraySize)
{
	clock_t start, end;

	start = clock();
	prefixAmount_cpu(&arr, arraySize);// закончить
	end = clock();

	float milliseconds = CALC_TIME_MS(start, end);
	printf("CPU: Time = %f ms\n", milliseconds);
	//printArray(arr, arraySize);
}

__host__ void GPU(ll* source, const size_t arraySize)
{
	ll* devSource = NULL, * result = NULL, * devResult = NULL;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float milliseconds = 0;
	const dim3 blockDim(MAX_TREAD_COUNT), gridDim((size_t)ceil(arraySize / ((double)blockDim.x)));

	///////////////////////////////////////GPU/////////////////////////////////////////////////////
	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaMalloc(&devSource, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devSource failed!");

	cudaStatus = cudaMalloc(&devResult, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMemcpy(devSource, source, arraySize * sizeof(ll), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devSource failed!");

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	cudaStatus = prefixAmount(&devSource, arraySize);
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

	//printArray(result, arraySize);
	printf("GPU time: %f ms\n", milliseconds);

Finish:
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
}


long main()
{
	const size_t arraySize = 1024 * 32;
	ll* arr1 = new ll[arraySize], * arr2 = new ll[arraySize];

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printDeviceProperties(deviceProp);

	fillArray(arr1, arraySize);
	//printArray(arr1, arraySize);

	memcpy(arr2, arr1, arraySize * sizeof(ll));

	CPU(arr1, arraySize);
	GPU(arr2, arraySize);

	delete[] arr1;
	delete[] arr2;
	system("pause");
	return 0;
}
#endif
