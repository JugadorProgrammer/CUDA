#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

// размер блока или размер подматрицы
#define ll long long
#define FINAL_MATRIX_HEIGHT 3.0
#define FINAL_MATRIX_WIDTH 2.0
#define MAX_2D_TREAD_COUNT 32.0
#define TILE_SIZE (size_t)MAX_2D_TREAD_COUNT
#define CALC_TIME_MS(start, end) (((double)((end) - (start)) * 1000.0) / CLOCKS_PER_SEC)

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

struct Size
{
	size_t width = 0;
	size_t height = 0;
};

__host__ void fillMatrix(ll* matrix, const struct Size size)
{
	for (size_t i = 0; i < size.width * size.height; ++i)
	{
		matrix[i] = i;
	}
}

__host__ void printMatrix(ll* matrix, const struct Size size, const char* matrixName)
{
	printf("Matrix %s:\n", matrixName);
	for (size_t i = 0; i < size.height; ++i)
	{
		for (size_t j = 0; j < size.width; ++j)
		{
			printf("%lld ", matrix[size.width * i + j]);
		}
		printf("\n");
	}
}

__host__ struct Size matrixMult(const ll* a, const ll* b, ll* result, struct Size aSize, struct Size bSize)
{
	struct Size resultSize;
	resultSize.width = bSize.width;
	resultSize.height = aSize.height;

	size_t n = aSize.width;
	for (size_t i = 0; i < resultSize.height; ++i)
	{
		for (size_t j = 0; j < resultSize.width; ++j)
		{
			size_t index = i * resultSize.width + j;
			result[index] = 0;
			for (size_t k = 0; k < n; ++k)
			{
				result[index] += a[i * aSize.width + k] * b[k * bSize.width + j];
			}
		}
	}

	return resultSize;
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

__global__ void matrixMultGPU(const ll* a, const ll* b, ll* result, struct Size* resultSize, struct Size aSize, struct Size bSize)
{
	size_t indexX = blockDim.x * blockIdx.x + threadIdx.x;
	size_t indexY = blockDim.y * blockIdx.y + threadIdx.y;

	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		resultSize->width = bSize.width;
		resultSize->height = aSize.height;
	}

	if (indexX >= bSize.width || indexY >= aSize.height)
	{
		return;
	}

	ll sum = 0ll;
	for (size_t k = 0; k < aSize.width; ++k)
	{
		sum += a[indexY * aSize.width + k] * b[k * bSize.width + indexX];
	}

	size_t index = bSize.width * indexY + indexX;
	result[index] = sum;
}

//__global__ void matrixMultGPUShared(const ll* a, const ll* b, ll* result, struct Size* resultSize, struct Size aSize, struct Size bSize)
//{
//	size_t indexX = blockDim.x * blockIdx.x + threadIdx.x;
//	size_t indexY = blockDim.y * blockIdx.y + threadIdx.y;
//
//	if (indexX >= bSize.width || indexY >= aSize.height)
//	{
//		return;
//	}
//
//	// индекс начала первой подматрицы А, которую
//	// обрабатывает блок
//	size_t aBegin = aSize.width * blockDim.y * blockIdx.y;
//	// индекс конца подматрицы А, которую обрабатывает блок
//	size_t aEnd = aBegin + aSize.width - 1;
//	// шаг для перебора подматриц А
//	size_t aStep = blockDim.x;
//	// индекс начала первой подматрицы В, которую
//	// обрабатывает блок
//	size_t bBegin = blockDim.x * blockIdx.x;
//	// шаг для перебора подматриц В
//	size_t bStep = blockDim.y * bSize.width;
//
//	// Выделение разделяемой памяти для подматриц
//	__shared__ ll as[TILE_SIZE][TILE_SIZE];
//	__shared__ ll bs[TILE_SIZE][TILE_SIZE];
//	// переменная для вычисления элемента подматрицы
//	ll sum = 0;
//	for (size_t ia = aBegin, ib = bBegin; ia < aEnd; ia += aStep, ib += bStep)
//	{
//		// загрузка подматриц А и В из глобальной памяти в
//		// разделяемую
//		as[threadIdx.y][threadIdx.x] = a[ia + aSize.width * threadIdx.y + threadIdx.x];
//		bs[threadIdx.y][threadIdx.x] = b[ib + bSize.width * threadIdx.y + threadIdx.x];
//		// синхронизация нитей
//		__syncthreads();
//		// перемножение двух матриц
//		for (size_t k = 0; k < blockDim.x; k++)
//		{
//			sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
//		}
//		// синхронизация нитей
//		__syncthreads();
//	}
//
//	size_t index = bSize.width * indexY + indexX;
//	result[index] = sum;
//
//	if (blockIdx.x == 0 && threadIdx.x == 0)
//	{
//		resultSize->width = bSize.width;
//		resultSize->height = aSize.height;
//	}
//}

__global__ void matrixMultGPUShared(const ll* a, const ll* b, ll* result, struct Size* resultSize, struct Size aSize, struct Size bSize)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
		resultSize->width = bSize.width;
		resultSize->height = aSize.height;
	}

	__shared__ ll tileA[TILE_SIZE][TILE_SIZE];
	__shared__ ll tileB[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	ll sum = 0;
	for (int tileIdx = 0; tileIdx < (aSize.width + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx)
	{
		// Загрузка tileA
		int aRow = blockIdx.y * TILE_SIZE + threadIdx.y;
		int aCol = tileIdx * TILE_SIZE + threadIdx.x;
		if (aRow < aSize.height && aCol < aSize.width)
		{
			tileA[threadIdx.y][threadIdx.x] = a[aRow * aSize.width + aCol];
		}
		else
		{
			tileA[threadIdx.y][threadIdx.x] = 0;
		}

		// Загрузка tileB  
		int bRow = tileIdx * TILE_SIZE + threadIdx.y;
		int bCol = blockIdx.x * TILE_SIZE + threadIdx.x;
		if (bRow < bSize.height && bCol < bSize.width)
		{
			tileB[threadIdx.y][threadIdx.x] = b[bRow * bSize.width + bCol];
		}
		else
		{
			tileB[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();

		// Вычисление частичной суммы
		for (int k = 0; k < TILE_SIZE; ++k) 
		{
			sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
		}

		__syncthreads();
	}

	// Запись результата
	if (row < resultSize->height && col < resultSize->width) 
	{
		result[row * resultSize->width + col] = sum;
	}
}

int main()
{
	srand(time(NULL));

	ll* a = NULL, * devA = NULL, * b = NULL, * devB = NULL, * resultCPU = NULL, * resultGPU = NULL, * devResult = NULL;
	struct Size aSize, bSize, * resultSize = NULL, * devResultSize = NULL;
	clock_t startCPU, endCPU;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaDeviceProp deviceProp;
	const dim3 blockDim(MAX_2D_TREAD_COUNT, MAX_2D_TREAD_COUNT), gridDim((size_t)ceil(FINAL_MATRIX_WIDTH / ((double)blockDim.x)), (size_t)ceil(FINAL_MATRIX_HEIGHT / ((double)blockDim.y)));
	float milliseconds = 0;

	aSize.width = FINAL_MATRIX_HEIGHT;
	aSize.height = FINAL_MATRIX_WIDTH;
	a = new ll[aSize.width * aSize.height];

	bSize.width = FINAL_MATRIX_WIDTH;
	bSize.height = FINAL_MATRIX_HEIGHT;
	b = new ll[bSize.width * bSize.height];

	fillMatrix(a, aSize);
	fillMatrix(b, bSize);

	printMatrix(a, aSize, "A");
	printMatrix(b, bSize, "B");

	resultCPU = new ll[aSize.height * bSize.width];

	resultSize = new Size();
	printf("CPU start calculation\n");
	startCPU = clock();
	(*resultSize) = matrixMult(a, b, resultCPU, aSize, bSize);
	endCPU = clock();
	milliseconds = CALC_TIME_MS(startCPU, endCPU);

	printMatrix(resultCPU, *resultSize, "CPU result");
	printf("CPU time: %f ms\n", milliseconds);

	///////////////////////////////////////GPU/////////////////////////////////////////////////////

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetDeviceProperties failed!");

	printDeviceProperties(deviceProp);
	cudaStatus = cudaMalloc(&devA, aSize.width * aSize.height * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devA failed!");

	cudaStatus = cudaMalloc(&devB, bSize.width * bSize.height * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devB failed!");

	cudaStatus = cudaMalloc(&devResult, aSize.height * bSize.width * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMalloc(&devResultSize, sizeof(Size));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultSize failed!");

	cudaStatus = cudaMemcpy(devA, a, aSize.width * aSize.height * sizeof(ll), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devA failed!");

	cudaStatus = cudaMemcpy(devB, b, bSize.width * bSize.height * sizeof(ll), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devB failed!");

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	matrixMultGPUShared << <gridDim, blockDim >> > (devA, devB, devResult, devResultSize, aSize, bSize);

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

	cudaStatus = cudaMemcpy(resultSize, devResultSize, sizeof(Size), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(&resultSize failed!");

	resultGPU = new ll[resultSize->width * resultSize->height];

	cudaStatus = cudaMemcpy(resultGPU, devResult, resultSize->width * resultSize->height * sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(resultGPU failed!");

	printMatrix(resultGPU, *resultSize, "GPU result");
	printf("GPU time: %f ms\n", milliseconds);

Finish:

	DELETE_ARRAY_IF_EXISTS(a);
	DELETE_ARRAY_IF_EXISTS(b);
	DELETE_ARRAY_IF_EXISTS(resultCPU);
	DELETE_ARRAY_IF_EXISTS(resultGPU);
	DELETE_IF_EXISTS(resultSize);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	if (devA)
	{
		cudaStatus = cudaFree(devA);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devA failed!");
	}

	if (devB)
	{
		cudaStatus = cudaFree(devB);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devB failed!");
	}

	if (devResult)
	{
		cudaStatus = cudaFree(devResult);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devResult failed!");
	}

	if (devResultSize)
	{
		cudaStatus = cudaFree(devResultSize);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devResultSize failed!");
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");

	system("pause");
	return 0;
}
#endif
