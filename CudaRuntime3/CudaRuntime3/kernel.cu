#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <chrono>
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

	/*memcpy(result, source + 1, (size - 1) * sizeof(ll));
	memcpy(result + (size - 1), &total_sum, sizeof(ll));
	*arr = result;*/
	printArray(*arr, size);
}

__global__ void prefixSumKernelShared(ll* input, ll* output, size_t n)
{
	extern __shared__ int temp[];

	int blockSize = blockDim.x;
	int tid = threadIdx.x;
	int blockId = blockIdx.x;
	int globalIdx = blockId * blockSize * 2 + tid;

	// Загрузка данных с двойной буферизацией для лучшей утилизации
	int offset = blockSize;
	temp[tid] = (globalIdx < n) ? input[globalIdx] : 0;
	temp[tid + offset] = (globalIdx + blockSize < n) ? input[globalIdx + blockSize] : 0;

	__syncthreads();

	// Восходящая фаза (Blelloch scan)
	for (int stride = 1; stride <= blockSize; stride *= 2) 
	{
		int index = (tid + 1) * stride * 2 - 1;
		if (index < blockSize * 2) 
		{
			temp[index] += temp[index - stride];
		}
		__syncthreads();
	}

	// Нисходящая фаза
	for (int stride = blockSize / 2; stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = (tid + 1) * stride * 2 - 1;
		if (index + stride < blockSize * 2) 
		{
			temp[index + stride] += temp[index];
		}
	}
	__syncthreads();

	// Сохранение результата
	if (globalIdx < n) 
	{
		output[globalIdx] = temp[tid];
	}
	if (globalIdx + blockSize < n) 
	{
		output[globalIdx + blockSize] = temp[tid + offset];
	}
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
	cudaStream_t stream;

	cudaStatus = cudaStreamCreate(&stream);
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	for (int stride = 1; stride < size; stride *= 2)
	{
		int num_threads_needed = (size + (2 * stride) - 1) / (2 * stride);
		int blocks_per_grid = (num_threads_needed + MAX_TREAD_COUNT - 1) / MAX_TREAD_COUNT;
		if (blocks_per_grid == 0)
		{
			blocks_per_grid = 1;
		}

		cudaStreamSynchronize(stream);
		upsweep_kernel << <blocks_per_grid, MAX_TREAD_COUNT, 0, stream >> > (source, size, stride);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			cudaStreamSynchronize(stream);
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

		cudaStreamSynchronize(stream);
		downsweep_kernel << <blocks_per_grid, MAX_TREAD_COUNT, 0, stream >> > (source, size, stride);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			cudaStreamSynchronize(stream);
			return cudaStatus;
		}
	}

	/*ll* result;
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

	*arr = result;*/
	return cudaSuccess;
}

__host__ void CPU(ll* arr, const size_t arraySize)
{
	auto start = std::chrono::high_resolution_clock::now();

	prefixAmount_cpu(&arr, arraySize);// закончить

	auto end = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration<double, std::milli>(end - start);
	printf("CPU: Time = %f ms\n", time.count());
}

__host__ void GPU(ll* source, const size_t arraySize)
{
	ll* devSource = NULL, * result = NULL;
	cudaError_t cudaStatus;
	cudaEvent_t memoryCopyStart, memoryCopyStop;
	cudaEvent_t start, stop;
	float milliseconds = 0;
	const dim3 blockDim(MAX_TREAD_COUNT), gridDim((size_t)ceil(arraySize / ((double)blockDim.x)));

	cudaStatus = cudaEventCreate(&memoryCopyStart);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&memoryCopyStart) failed!");

	cudaStatus = cudaEventCreate(&memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&memoryCopyStop) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventRecord(memoryCopyStart);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&memoryCopyStart) failed!");

	cudaStatus = cudaMalloc(&devSource, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devSource failed!");

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

	printf("GPU calculate: %f ms\n", milliseconds);

	cudaStatus = cudaEventRecord(memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&memoryCopyStop) failed!");

	// Ждем завершения всех операций
	cudaStatus = cudaEventSynchronize(memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventSynchronize(&memoryCopyStop) failed!");

	cudaStatus = cudaEventElapsedTime(&milliseconds, memoryCopyStart, memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventElapsedTime failed!");

	printf("GPU time: %f ms\n", milliseconds);


	printArray(result, arraySize);
Finish:
	DELETE_ARRAY_IF_EXISTS(result);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaEventDestroy(memoryCopyStart);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(memoryCopyStart failed!");

	cudaStatus = cudaEventDestroy(memoryCopyStop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(memoryCopyStop failed!");

	if (devSource)
	{
		cudaStatus = cudaFree(devSource);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devSource failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ void GPUShared(ll* source, const size_t arraySize)
{
	ll* devSource = NULL, * result = NULL, * devResult = NULL;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEvent_t memoryCopyStart, memoryCopyStop;
	float milliseconds = 0;
	const dim3 blockDim(MAX_TREAD_COUNT), gridDim((size_t)ceil(arraySize / ((double)blockDim.x)));

	cudaStatus = cudaEventCreate(&memoryCopyStart);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&memoryCopyStart) failed!");

	cudaStatus = cudaEventCreate(&memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&memoryCopyStop) failed!");

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaEventRecord(memoryCopyStart);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&memoryCopyStart) failed!");

	cudaStatus = cudaMalloc(&devSource, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devSource failed!");

	cudaStatus = cudaMalloc(&devResult, arraySize * sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResult failed!");

	cudaStatus = cudaMemcpy(devSource, source, arraySize * sizeof(ll), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devSource failed!");

	printf("Shared GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	int numBlocks = (arraySize + MAX_TREAD_COUNT - 1) / MAX_TREAD_COUNT;
	int sharedMemSize = MAX_TREAD_COUNT * sizeof(ll);

	prefixSumKernelShared << <numBlocks, MAX_TREAD_COUNT, sharedMemSize >> > (devSource, devResult, arraySize);
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
	cudaStatus = cudaMemcpy(result, devResult, arraySize * sizeof(ll), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(&result failed!");

	cudaStatus = cudaEventRecord(memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&memoryCopyStop) failed!");

	printf("Shared GPU calculate: %f ms\n", milliseconds);
	// Ждем завершения всех операций
	cudaStatus = cudaEventSynchronize(memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventSynchronize(&memoryCopyStop) failed!");

	cudaStatus = cudaEventElapsedTime(&milliseconds, memoryCopyStart, memoryCopyStop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventElapsedTime failed!");

	printf("Shared GPU time: %f ms\n", milliseconds); 

	memcpy(source + 1, result, (arraySize - 1) * sizeof(ll));
	source[0] = 0;
	printArray(source, arraySize);

Finish:
	DELETE_ARRAY_IF_EXISTS(result);

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaEventDestroy(memoryCopyStart);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(memoryCopyStart failed!");

	cudaStatus = cudaEventDestroy(memoryCopyStop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(memoryCopyStop failed!");

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
	const size_t arraySize = 8;
	ll* arr1 = new ll[arraySize], * arr2 = new ll[arraySize], * arr3 = new ll[arraySize];

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printDeviceProperties(deviceProp);

	fillArray(arr1, arraySize);
	
	memcpy(arr2, arr1, arraySize * sizeof(ll));
	memcpy(arr3, arr1, arraySize * sizeof(ll));

	CPU(arr1, arraySize);
	GPU(arr2, arraySize);
	GPUShared(arr3, arraySize);

	/*printArray(arr1, arraySize);
	printArray(arr2, arraySize);
	printArray(arr3, arraySize);*/

	delete[] arr1;
	delete[] arr2;
	delete[] arr3;
	system("pause");
	return 0;
}
#endif
