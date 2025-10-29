#pragma once
#ifndef __longELLISENSE_
#define NOMINMAX

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <thread>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define ll float
#define minimum(a, b) (((a) < (b)) ? (a) : (b))
#define TILE_SIZE (size_t)MAX_2D_TREAD_COUNT
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
		printf("%f ", arr[i]);
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

__host__ ll parallel_reduction(ll *array, size_t size, int num_threads)
{
	// Ограничиваем количество потоков размером массива
	num_threads = std::min(num_threads, static_cast<int>(size));

	// Функция для работы потока
	auto worker = [&](int thread_id) 
	{
		int chunk_size = size / num_threads;
		int start = thread_id * chunk_size;
		int end = (thread_id == num_threads - 1) ? size : start + chunk_size;

		// Локальная редукция для каждого потока
		float local_sum = 0.0f;
		for (int i = start; i < end; ++i) 
		{
			local_sum += array[i];
		}
		array[thread_id] = local_sum; // Сохраняем результат в начало массива
	};

	// Запускаем потоки для первой фазы
	std::thread* threads = new std::thread[num_threads];
	for (int i = 0; i < num_threads; ++i) 
	{
		threads[i] = std::thread(worker, i);
	}

	// Ждем завершения всех потоков
	for (int i = 0; i < num_threads; ++i)
	{
		threads[i].join();
	}
	delete[] threads;

	// Древовидная редукция результатов
	int remaining = num_threads;
	while (remaining > 1) 
	{
		int half = (remaining + 1) / 2;

		std::thread* merge_threads = new std::thread[half];
		for (int i = 0; i < half; ++i) 
		{
			int second_idx = i + half;
			if (second_idx < remaining)
			{
				merge_threads[i] = std::thread([&, i, second_idx]() 
					{
						array[i] += array[second_idx];
					});
			}
			else {
				// Если нет пары, просто копируем значение
				merge_threads[i] = std::thread([&, i]() 
					{
					// Значение уже на своем месте
					});
			}
		}

		for (int i = 0; i < half; ++i) {
			merge_threads[i].join();
		}
		delete[] merge_threads;

		remaining = half;
	}

	return array[0];
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
	auto start = std::chrono::high_resolution_clock::now();

	ll result = parallel_reduction(arr, arraySize, MAX_CPU_LOGICAL_THREAD);

	auto end = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration<double, std::milli>(end - start);
	printf("CPU: Result = %f Time = %f ms\n", result, time.count());
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

	// Создаем CUDA stream для асинхронной обработки
	cudaStream_t stream;
	cudaStatus = cudaStreamCreate(&stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate failed!");

	// Инициализируем переменные
	cudaStatus = cudaMallocAsync(&devArr, byteSize, stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc(&devResult, sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpyAsync(devArr, arr, byteSize, cudaMemcpyHostToDevice, stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	cudaStreamSynchronize(stream);
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

	printf("GPU: Result = %f Time = %f ms\n", *result, milliseconds);
Finish:
	DELETE_IF_EXISTS(result);
	// Освобождаем ресурсы
	cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaStreamDestroy(stream);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy failed!");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ void GPUShared(ll* arr, const size_t arraySize)
{
	ll* result = NULL, * devResult = NULL, * devArr = NULL;
	size_t byteSize = arraySize * sizeof(ll);

	cudaError_t cudaStatus;
	cudaStream_t stream;
	cudaEvent_t start, stop;
	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate failed!");

	cudaStatus = cudaStreamCreate(&stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate failed!");
	// Инициализируем переменные
	cudaStatus = cudaMallocAsync(&devArr, byteSize, stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMalloc(&devResult, sizeof(ll));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpyAsync(devArr, arr, byteSize, cudaMemcpyHostToDevice, stream);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy failed!");

	cudaStreamSynchronize(stream);
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

	printf("GPU_SHARED: Result = %f Time = %f ms\n", *result, milliseconds);
Finish:
	DELETE_IF_EXISTS(result);
	// Освобождаем ресурсы
	cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaStreamDestroy(stream);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy failed!");

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ void printDeviceProperties(const cudaDeviceProp& deviceProp)
{
	// ============ ОСНОВНАЯ ИНФОРМАЦИЯ О GPU ============
	printf("\n\nGPU: %s\n", deviceProp.name);  // Название графического процессора
	printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);  // Версия вычислительной возможности
	printf("Global Memory: %.2f GB\n", deviceProp.totalGlobalMem / 1073741824.0);  // Общий объем глобальной памяти в GB
	printf("Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);  // Ширина шины памяти в битах
	printf("Memory Clock Rate: %.2f GHz\n", deviceProp.memoryClockRate * 1e-6f);  // Тактовая частота памяти в GHz

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

long main()
{
	srand(time(NULL));
	const size_t arraySize = 2 << 20;
	ll* arr = new ll[arraySize], * arr1 = new ll[arraySize];

	cudaDeviceProp deviceProp;
	PRINT_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0), "cudaGetDeviceProperties failed!");
	printDeviceProperties(deviceProp);

	fillArray(arr, arraySize);
	//printArray(arr, arraySize);
	memcpy(arr1, arr, arraySize * sizeof(ll));

	CPU(arr, arraySize);
	GPU(arr1, arraySize);
	GPUShared(arr1, arraySize);

	delete[] arr;
	system("pause");
	return 0;
}
#endif