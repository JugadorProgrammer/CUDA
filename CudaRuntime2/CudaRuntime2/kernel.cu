#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>

// размер блока или размер подматрицы
#define ll float
#define FINAL_MATRIX_HEIGHT 1024.0
#define FINAL_MATRIX_WIDTH 1024.0
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
			printf("%f ", matrix[size.width * i + j]);
		}
		printf("\n");
	}
}

__host__ struct Size matrixMult(const ll* a, const ll* b, ll** result, struct Size aSize, struct Size bSize)
{
	struct Size resultSize;
	resultSize.width = bSize.width;
	resultSize.height = aSize.height;
	(*result) = new ll[resultSize.width * resultSize.height];

	size_t n = aSize.width;
	for (size_t i = 0; i < resultSize.height; ++i)
	{
		for (size_t j = 0; j < resultSize.width; ++j)
		{
			size_t index = i * resultSize.width + j;
			(*result)[index] = 0;
			for (size_t k = 0; k < n; ++k)
			{
				(*result)[index] += a[i * aSize.width + k] * b[k * bSize.width + j];
			}
		}
	}

	return resultSize;
}

__host__ void printDeviceProperties(const cudaDeviceProp& deviceProp)
{
	// ============ ОСНОВНАЯ ИНФОРМАЦИЯ О GPU ============
	printf("\n\nGPU: %s\n", deviceProp.name);  // Название графического процессора
	printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);  // Версия вычислительной возможности
	printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / 1073741824.0);  // Общий объем глобальной памяти в GB
	printf("Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);  // Ширина шины памяти в битах
	printf("Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6f);  // Тактовая частота памяти в GHz

	// Расчет теоретической пиковой пропускной способности памяти
	float memoryBandwidth = 2.0f * deviceProp.memoryClockRate * 1e3f *
		(deviceProp.memoryBusWidth / 8) / 1e9f;
	printf("Theoretical Memory Bandwidth: %.2f GB/s\n", memoryBandwidth);

	// ============ СТРУКТУРА БЛОКОВ И СЕТКИ ============
	printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);  // Максимальное количество потоков в одном блоке
	printf("Max Block Dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);  // Максимальные размеры блока по осям X, Y, Z
	printf("Max Grid Dim: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);  // Максимальные размеры сетки по осям X, Y, Z

	// ============ АППАРАТНЫЕ ХАРАКТЕРИСТИКИ ============
	printf("Max blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);  // Максимальное количество блоков на одном мультипроцессоре
	printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);  // Количество мультипроцессоров (Streaming Multiprocessors - SM)
	printf("Clock Rate: %.2f GHz\n", deviceProp.clockRate * 1e-6f);  // Тактовая частота ядер GPU в GHz
	printf("Shared Memory per Block: %zu KB\n", deviceProp.sharedMemPerBlock / 1024);  // Объем разделяемой памяти на блок в KB
	printf("Shared Memory per Multiprocessor: %zu KB\n", deviceProp.sharedMemPerMultiprocessor / 1024);  // Общий объем разделяемой памяти на мультипроцессор в KB
	printf("Registers per Block: %d\n", deviceProp.regsPerBlock);  // Количество 32-битных регистров на блок
	printf("Registers per Multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);  // Общее количество регистров на мультипроцессоре

	// Расчет теоретической производительности в FLOPS
	// Вспомогательная функция для определения количества ядер на мультипроцессор
	auto _ConvertSMVer2Cores = [](int major, int minor) -> int {
		struct SMVersion { int major, minor, cores; };
		SMVersion smVersions[] = {
			{3, 0, 192}, {3, 5, 192}, {3, 7, 192},  // Kepler
			{5, 0, 128}, {5, 2, 128}, {5, 3, 128},  // Maxwell
			{6, 0, 64}, {6, 1, 128}, {6, 2, 128},   // Pascal
			{7, 0, 64}, {7, 2, 64}, {7, 5, 64},     // Volta, Turing
			{8, 0, 64}, {8, 6, 128}, {8, 9, 128},   // Ampere, Ada Lovelace
			{9, 0, 128}                              // Hopper
		};
		for (const auto& sm : smVersions) {
			if (sm.major == major && sm.minor == minor) {
				return sm.cores;
			}
		}
		return 128;  // Значение по умолчанию
	};

	float totalCores = deviceProp.multiProcessorCount *
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);  // Общее количество CUDA ядер
	float theoreticalFlops = totalCores * deviceProp.clockRate * 1e3f * 2;  // Теоретическая производительность FP32
	printf("Theoretical FP32 Performance: %.2f GFLOPS\n", theoreticalFlops / 1e9f);  // Вывод в GFLOPS

	// ============ ПАРАЛЛЕЛИЗМ И ВОЗМОЖНОСТИ ============
	printf("Warp Size: %d\n", deviceProp.warpSize);  // Размер warp'а (основная единица выполнения)
	printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);  // Максимальное количество потоков на мультипроцессоре
	printf("Concurrent Kernels: %s\n", deviceProp.concurrentKernels ? "Yes" : "No");  // Поддержка одновременного выполнения нескольких ядер
	printf("Concurrent Copy/Execute: %s\n", deviceProp.deviceOverlap ? "Yes" : "No");  // Поддержка перекрытия копирования данных и вычислений
	printf("Integrated GPU: %s\n", deviceProp.integrated ? "Yes" : "No");  // Является ли GPU интегрированным

	// ============ КЭШ ПАМЯТЬ ============
	printf("L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);  // Размер L2 кэша в KB
	printf("Persisting L2 Cache Max Size: %zu KB\n", deviceProp.persistingL2CacheMaxSize / 1024);  // Максимальный размер persistent L2 кэша в KB

	// ============ ПОДДЕРЖКА РАЗЛИЧНЫХ ФУНКЦИЙ ============
	printf("Unified Addressing: %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");  // Единое адресное пространство для CPU и GPU
	printf("Managed Memory: %s\n", deviceProp.managedMemory ? "Yes" : "No");  // Поддержка managed memory
	printf("Compute Preemption: %s\n", deviceProp.computePreemptionSupported ? "Yes" : "No");  // Поддержка вытеснения вычислений
	printf("Cooperative Launch: %s\n", deviceProp.cooperativeLaunch ? "Yes" : "No");  // Поддержка cooperative launch

	// ============ ПОДДЕРЖКА АТОМАРНЫХ ОПЕРАЦИЙ ============
	printf("Host Native Atomic Supported: %s\n", deviceProp.hostNativeAtomicSupported ? "Yes" : "No");  // Поддержка атомарных операций на host памяти
	printf("Single To Double Precision Perf Ratio: %d\n", deviceProp.singleToDoublePrecisionPerfRatio);  // Соотношение производительности single/double precision

	// ============ PCIe ИНФОРМАЦИЯ ============
	printf("PCI Bus ID: %d\n", deviceProp.pciBusID);  // ID PCIe шины
	printf("PCI Device ID: %d\n", deviceProp.pciDeviceID);  // ID PCIe устройства
	printf("PCI Domain ID: %d\n", deviceProp.pciDomainID);  // ID PCIe домена

	// ============ ПОДДЕРЖКА ECC И ДРАЙВЕРОВ ============
	printf("ECC Enabled: %s\n", deviceProp.ECCEnabled ? "Yes" : "No");  // Включена ли коррекция ошибок (ECC)
	printf("TCC Driver: %s\n\n\n", deviceProp.tccDriver ? "Yes" : "No");  // Используется ли TCC драйвер (Tesla Compute Cluster)
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

__host__ void CPU(ll* a, ll* b, const Size& aSize, const Size& bSize)
{
	printf("CPU start calculation\n");
	ll* resultCPU = new ll[0];
	
	auto start = std::chrono::high_resolution_clock::now();

	Size resultSize = matrixMult(a, b, &resultCPU, aSize, bSize);

	auto end = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration<double, std::milli>(end - start);
	printf("CPU: Time = %f ms\n", time.count());
	//printMatrix(resultCPU, resultSize, "CPU result");
	delete[] resultCPU;
}

__host__ void GPU(ll* a, ll* b, const Size& aSize, const Size& bSize)
{
	ll* resultGPU = NULL, * devA = NULL, * devB = NULL, * devResult = NULL, * result = NULL;
	struct Size* resultSize = NULL, * devResultSize = NULL;

	cudaStream_t streamA, streamB;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	const dim3 blockDim(MAX_2D_TREAD_COUNT, MAX_2D_TREAD_COUNT), gridDim((size_t)ceil(FINAL_MATRIX_WIDTH / ((double)blockDim.x)), (size_t)ceil(FINAL_MATRIX_HEIGHT / ((double)blockDim.y)));
	float milliseconds = 0;

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaStreamCreate(&streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate(streamA failed!");

	cudaStatus = cudaStreamCreate(&streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate(stereamB failed!");

	cudaStatus = cudaMallocAsync(&devA, aSize.width * aSize.height * sizeof(ll), streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devA failed!");

	cudaStatus = cudaMallocAsync(&devB, bSize.width * bSize.height * sizeof(ll), streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devB failed!");

	cudaStatus = cudaMallocAsync(&devResult, aSize.height * bSize.width * sizeof(ll), streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMallocAsync(&devResultSize, sizeof(Size), streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultSize failed!");

	cudaStatus = cudaMemcpyAsync(devA, a, aSize.width * aSize.height * sizeof(ll), cudaMemcpyHostToDevice, streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devA failed!");

	cudaStatus = cudaMemcpyAsync(devB, b, bSize.width * bSize.height * sizeof(ll), cudaMemcpyHostToDevice, streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devB failed!");

	cudaStreamSynchronize(streamA);
	cudaStreamSynchronize(streamB);

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	matrixMultGPU<<<gridDim, blockDim>>>(devA, devB, devResult, devResultSize, aSize, bSize);

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

	resultSize = new Size();
	cudaStatus = cudaMemcpy(resultSize, devResultSize, sizeof(Size), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(&resultSize failed!");

	resultGPU = new ll[resultSize->width * resultSize->height];

	cudaStatus = cudaMemcpy(resultGPU, devResult, resultSize->width * resultSize->height * sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(resultGPU failed!");

	//printMatrix(resultGPU, *resultSize, "\nGPU result");
	printf("GPU time: %f ms\n", milliseconds);

Finish:
	DELETE_ARRAY_IF_EXISTS(resultGPU);
	DELETE_IF_EXISTS(resultSize);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaStreamDestroy(streamA);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(stereamA failed!");

	cudaStatus = cudaStreamDestroy(streamB);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(stereamB failed!");

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
}

__host__ void GPUShared(ll* a, ll* b, const Size& aSize, const Size& bSize)
{
	ll* resultGPU = NULL, * devA = NULL, * devB = NULL, * devResult = NULL, * result = NULL;
	struct Size* resultSize = NULL, * devResultSize = NULL;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaStream_t streamA, streamB;
	const dim3 blockDim(MAX_2D_TREAD_COUNT, MAX_2D_TREAD_COUNT), gridDim((size_t)ceil(FINAL_MATRIX_WIDTH / ((double)blockDim.x)), (size_t)ceil(FINAL_MATRIX_HEIGHT / ((double)blockDim.y)));
	float milliseconds = 0;

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaStreamCreate(&streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate(streamA failed!");

	cudaStatus = cudaStreamCreate(&streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate(stereamB failed!");

	cudaStatus = cudaMallocAsync(&devA, aSize.width * aSize.height * sizeof(ll), streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devA failed!");

	cudaStatus = cudaMallocAsync(&devB, bSize.width * bSize.height * sizeof(ll), streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devB failed!");

	cudaStatus = cudaMallocAsync(&devResult, aSize.height * bSize.width * sizeof(ll), streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMallocAsync(&devResultSize, sizeof(Size), streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultSize failed!");

	cudaStatus = cudaMemcpyAsync(devA, a, aSize.width * aSize.height * sizeof(ll), cudaMemcpyHostToDevice, streamA);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devA failed!");

	cudaStatus = cudaMemcpyAsync(devB, b, bSize.width * bSize.height * sizeof(ll), cudaMemcpyHostToDevice, streamB);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devB failed!");

	cudaStreamSynchronize(streamA);
	cudaStreamSynchronize(streamB);

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	matrixMultGPUShared<<<gridDim, blockDim>>>(devA, devB, devResult, devResultSize, aSize, bSize);

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

	resultSize = new Size();
	cudaStatus = cudaMemcpy(resultSize, devResultSize, sizeof(Size), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(&resultSize failed!");

	resultGPU = new ll[resultSize->width * resultSize->height];

	cudaStatus = cudaMemcpy(resultGPU, devResult, resultSize->width * resultSize->height * sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(resultGPU failed!");

	//printMatrix(resultGPU, *resultSize, "\nGPUShared result");
	printf("GPUShared time: %f ms\n", milliseconds);

Finish:
	DELETE_ARRAY_IF_EXISTS(resultGPU);
	DELETE_IF_EXISTS(resultSize);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaStreamDestroy(streamA);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(stereamA failed!");

	cudaStatus = cudaStreamDestroy(streamB);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(stereamB failed!");

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
}

long main()
{
	srand(time(NULL));
	ll* a = NULL, * b = NULL;
	struct Size aSize, bSize;

	aSize.width = FINAL_MATRIX_HEIGHT;
	aSize.height = FINAL_MATRIX_WIDTH;
	a = new ll[aSize.width * aSize.height];

	bSize.width = FINAL_MATRIX_WIDTH;
	bSize.height = FINAL_MATRIX_HEIGHT;
	b = new ll[bSize.width * bSize.height];

	fillMatrix(a, aSize);
	fillMatrix(b, bSize);

	/*printMatrix(a, aSize, "A");
	printMatrix(b, bSize, "B");*/

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printDeviceProperties(deviceProp);

	CPU(a, b, aSize, bSize);
	GPU(a, b, aSize, bSize);
	GPUShared(a, b, aSize, bSize);

	delete[] a;
	delete[] b;
	system("pause");
	return 0;
}
#endif