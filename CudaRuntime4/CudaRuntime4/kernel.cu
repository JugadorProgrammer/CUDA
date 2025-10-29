#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>

#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define ll long long
#define TILE_SIZE 32
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

__host__ void printDeviceProperties(const cudaDeviceProp & deviceProp)
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

__global__ void gaussianBlurKernelShared(uchar* image, uchar* output, int width, int height, float* kernel, size_t kernelSize)
{
	int tileStartX = blockIdx.x * blockDim.x;
	int tileStartY = blockIdx.y * blockDim.y;
	int tileEndX = (blockIdx.x + 1) * blockDim.x;
	int tileEndY = (blockIdx.y + 1) * blockDim.y;

	int x = tileStartX + threadIdx.x;
	int y = tileStartY + threadIdx.y;

	// Проверка выхода за границы изображения
	if (x < 2 || y < 2 || x > width - 2 || y > height - 2)
	{
		return;
	}

	int radius = kernelSize / 2;
	float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f, sumA = 0.0f;
	__shared__ uchar tile[TILE_SIZE * TILE_SIZE * 4];

	tile[(y % TILE_SIZE * TILE_SIZE + x % TILE_SIZE) * 4] = image[(y * width + x) * 4];
	tile[(y % TILE_SIZE * TILE_SIZE + x % TILE_SIZE) * 4 + 1] = image[(y * width + x) * 4 + 1];
	tile[(y % TILE_SIZE * TILE_SIZE + x % TILE_SIZE) * 4 + 2] = image[(y * width + x) * 4 + 2];
	tile[(y % TILE_SIZE * TILE_SIZE + x % TILE_SIZE) * 4 + 3] = image[(y * width + x) * 4 + 3];
	__syncthreads();
	// Применяем ядро Гаусса к каждому каналу BGRA
	for (int ky = -radius; ky <= radius; ++ky)
	{
		for (int kx = -radius; kx <= radius; ++kx)
		{
			int posX = x + kx;
			int posY = y + ky;

			// Безопасная обработка граничных условий (clamp to edge)
			if (posX > width - 1)
			{
				posX = width - 1;
			}
			if (posY > height - 1)
			{
				posY = height - 1;
			}

			int kernelIndex = (ky + radius) * kernelSize + (kx + radius);
			float kernelValue = kernel[kernelIndex];
			if (posX < tileStartX || posX >= tileEndX || posY < tileStartY || posY >= tileEndY)
			{
				int pixelIndex = (posY * width + posX) * 4;
				//// Умножаем каждый канал на коэффициент ядра
				sumB += image[pixelIndex] * kernelValue;     // Blue
				sumG += image[pixelIndex + 1] * kernelValue; // Green
				sumR += image[pixelIndex + 2] * kernelValue; // Red
				sumA += image[pixelIndex + 3] * kernelValue; // Alpha
				continue;
			}

			int pixelIndex = ((posY % TILE_SIZE) * TILE_SIZE + posX % TILE_SIZE) * 4;
			//// Умножаем каждый канал на коэффициент ядра
			sumB += tile[pixelIndex] * kernelValue;     // Blue
			sumG += tile[pixelIndex + 1] * kernelValue; // Green
			sumR += tile[pixelIndex + 2] * kernelValue; // Red
			sumA += tile[pixelIndex + 3] * kernelValue; // Alpha
		}
	}

	int outputIndex = (y * width + x) * 4;
	output[outputIndex] = (uchar)(sumB);     // Blue
	output[outputIndex + 1] = (uchar)(sumG); // Green
	output[outputIndex + 2] = (uchar)(sumR); // Red
	output[outputIndex + 3] = (uchar)(sumA); // Alpha
}

__global__ void gaussianBlurKernel(uchar* image, uchar* output, int width, int height, float* kernel, size_t kernelSize)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	// Проверка выхода за границы изображения
	if (x < 2 || y < 2 || x > width - 2 || y > height - 2)
	{
		return;
	}

	int radius = kernelSize / 2;
	float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f, sumA = 0.0f;

	// Применяем ядро Гаусса к каждому каналу BGRA
	for (int ky = -radius; ky <= radius; ++ky)
	{
		for (int kx = -radius; kx <= radius; ++kx)
		{
			int posX = x + kx;
			int posY = y + ky;

			// Безопасная обработка граничных условий (clamp to edge)
			if (posX > width - 1)
			{
				posX = width - 1;
			}
			if (posY > height - 1)
			{
				posY = height - 1;
			}

			// Получаем индекс пикселя в BGRA формате (4 канала на пиксель)
			int pixelIndex = (posY * width + posX) * 4;
			int kernelIndex = (ky + radius) * kernelSize + (kx + radius);
			float kernelValue = kernel[kernelIndex];

			//// Умножаем каждый канал на коэффициент ядра
			sumB += image[pixelIndex] * kernelValue;     // Blue
			sumG += image[pixelIndex + 1] * kernelValue; // Green
			sumR += image[pixelIndex + 2] * kernelValue; // Red
			sumA += image[pixelIndex + 3] * kernelValue; // Alpha
		}
	}

	int outputIndex = (y * width + x) * 4;
	output[outputIndex] = (uchar)(sumB);     // Blue
	output[outputIndex + 1] = (uchar)(sumG); // Green
	output[outputIndex + 2] = (uchar)(sumR); // Red
	output[outputIndex + 3] = (uchar)(sumA); // Alpha
}

__host__ void gaussianBlurKernelCPU(uchar* image, uchar* output, int width, int height, const float* kernel, const size_t kernelSize)
{
	const int radius = kernelSize / 2;
	// Создаем временный буфер
	uchar* temp = new uchar[width * height];
	memcpy(temp, image, width * height);

	// Применяем фильтр по горизонтали
	for (int y = 0; y < height; ++y) 
	{
		for (int x = 0; x < width; ++x) 
		{
			if (x < 2 || y < 2 || x > width - 2 || y > height - 2)
			{
				continue;
			}

			float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f, sumA = 0.0f;

			// Применяем ядро Гаусса к каждому каналу BGRA
			for (int ky = -radius; ky <= radius; ++ky)
			{
				for (int kx = -radius; kx <= radius; ++kx)
				{
					int posX = x + kx;
					int posY = y + ky;

					// Безопасная обработка граничных условий (clamp to edge)
					if (posX > width - 1)
					{
						posX = width - 1;
					}
					if (posY > height - 1)
					{
						posY = height - 1;
					}

					// Получаем индекс пикселя в BGRA формате (4 канала на пиксель)
					int pixelIndex = (posY * width + posX) * 4;
					int kernelIndex = (ky + radius) * kernelSize + (kx + radius);
					float kernelValue = kernel[kernelIndex];

					//// Умножаем каждый канал на коэффициент ядра
					sumB += image[pixelIndex] * kernelValue;     // Blue
					sumG += image[pixelIndex + 1] * kernelValue; // Green
					sumR += image[pixelIndex + 2] * kernelValue; // Red
					sumA += image[pixelIndex + 3] * kernelValue; // Alpha
				}
			}

			int outputIndex = (y * width + x) * 4;
			output[outputIndex] = (uchar)(sumB);     // Blue
			output[outputIndex + 1] = (uchar)(sumG); // Green
			output[outputIndex + 2] = (uchar)(sumR); // Red
			output[outputIndex + 3] = (uchar)(sumA); // Alpha
		}
	}
}

__host__ void CPU(const cv::Mat& image, const float* kernel, const size_t kernelSize)
{
	cv::Mat resultMat = cv::Mat(image.rows, image.cols, image.type());

	printf("CPU start calculation\n");
	clock_t start, end;

	start = clock();
	gaussianBlurKernelCPU(image.data, resultMat.data, image.cols, image.rows, kernel, kernelSize);
	end = clock();

	float milliseconds = CALC_TIME_MS(start, end);
	printf("CPU: Time = %f ms\n", milliseconds);
	cv::imwrite("СPU_Result.png", resultMat);
}

__host__ void GPU(const cv::Mat& image, const float* kernel, const size_t kernelSize)
{
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaStream_t streamImage, streamKernel;
	int width = image.cols, height = image.rows, channelsCount = image.channels();

	float* devKernel = NULL, milliseconds = 0;
	cv::Mat resultMat;
	uchar* devImage = NULL, * devResultImage = NULL;
	const dim3 blockDim(TILE_SIZE, TILE_SIZE),
		gridDim((size_t)std::ceil((double)width / (double)TILE_SIZE), (size_t)std::ceil((double)height / (double)TILE_SIZE));

	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaStreamCreate(&streamImage);
	CHECK_CUDA_ERROR(cudaStatus, "cudaStreamCreate failed!");

	cudaStatus = cudaMallocAsync(&devImage, width * height * channelsCount * sizeof(uchar), streamImage);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devImage failed!");

	cudaStatus = cudaMallocAsync(&devResultImage, width * height * channelsCount * sizeof(uchar), streamImage);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultImage failed!");

	cudaStatus = cudaMallocAsync(&devKernel, kernelSize * kernelSize * sizeof(float), streamKernel);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devKernel failed!");

	cudaStatus = cudaMemcpyAsync(devKernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice, streamKernel);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devKernel failed!");

	cudaStatus = cudaMemcpyAsync(devImage, image.data, width * height * channelsCount * sizeof(uchar), cudaMemcpyHostToDevice, streamImage);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devImage failed!");

	cudaStreamSynchronize(streamImage);
	cudaStreamSynchronize(streamKernel);
	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	gaussianBlurKernel << <gridDim, blockDim >> > (devImage, devResultImage, width, height, devKernel, kernelSize);

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

	resultMat = cv::Mat(image.rows, image.cols, image.type());
	cudaStatus = cudaMemcpy(resultMat.data, devResultImage, width * height * channelsCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devResultImage failed!");

	printf("GPU time: %f ms\n", milliseconds);

	cv::imwrite("GPU_Result.png", resultMat);
Finish:
	resultMat.release();

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	cudaStatus = cudaStreamDestroy(streamImage);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(streamImage failed!");

	cudaStatus = cudaStreamDestroy(streamKernel);
	PRINT_CUDA_ERROR(cudaStatus, "cudaStreamDestroy(streamKernel failed!");

	if (devImage)
	{
		cudaStatus = cudaFree(devImage);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devImage failed!");
	}

	if (devResultImage)
	{
		cudaStatus = cudaFree(devResultImage);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devResultImage failed!");
	}

	if (devKernel)
	{
		cudaStatus = cudaFree(devKernel);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devKernel failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ void GPUShared(const cv::Mat& image, const float* kernel, const size_t kernelSize)
{
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	int width = image.cols, height = image.rows, channelsCount = image.channels();

	float* devKernel = NULL, milliseconds = 0;
	cv::Mat resultMat;
	uchar* devImage = NULL, * devResultImage = NULL;
	const dim3 blockDim(TILE_SIZE, TILE_SIZE),
		gridDim((size_t)std::ceil((double)width / (double)TILE_SIZE), (size_t)std::ceil((double)height / (double)TILE_SIZE));

	///////////////////////////////////////GPU/////////////////////////////////////////////////////
	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaMalloc(&devImage, width * height * channelsCount * sizeof(uchar));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devImage failed!");

	cudaStatus = cudaMalloc(&devResultImage, width * height * channelsCount * sizeof(uchar));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultImage failed!");

	cudaStatus = cudaMalloc(&devKernel, kernelSize * kernelSize * sizeof(float));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devKernel failed!");

	cudaStatus = cudaMemcpy(devKernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devKernel failed!");

	cudaStatus = cudaMemcpy(devImage, image.data, width * height * channelsCount * sizeof(uchar), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devImage failed!");

	printf("Shared GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	gaussianBlurKernelShared << <gridDim, blockDim >> > (devImage, devResultImage, width, height, devKernel, kernelSize);

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

	resultMat = cv::Mat(image.rows, image.cols, image.type());
	cudaStatus = cudaMemcpy(resultMat.data, devResultImage, width * height * channelsCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devResultImage failed!");

	printf("GPU Shared time: %f ms\n", milliseconds);

	cv::imwrite("Shared_Result.png", resultMat);
Finish:
	resultMat.release();

	// Освобождаем ресурсы
	cudaStatus = cudaEventDestroy(start);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(start failed!");

	cudaStatus = cudaEventDestroy(stop);
	PRINT_CUDA_ERROR(cudaStatus, "cudaEventDestroy(stop failed!");

	if (devImage)
	{
		cudaStatus = cudaFree(devImage);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devImage failed!");
	}

	if (devResultImage)
	{
		cudaStatus = cudaFree(devResultImage);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devResultImage failed!");
	}

	if (devKernel)
	{
		cudaStatus = cudaFree(devKernel);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devKernel failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");
}

__host__ float* createKernel(size_t kernelSize, float sigma = 1.0)
{
	kernelSize |= 1; // должен быть нечётным
	float* kernel = new float[kernelSize * kernelSize], sum = 0;
	int radius = kernelSize / 2;

	for (int y = -radius; y <= radius; ++y)
	{
		for (int x = -radius; x <= radius; ++x)
		{
			float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel[(y + radius) * kernelSize + (x + radius)] = value;
			sum += value;
		}
	}

	// Нормализация
	for (int i = 0; i < kernelSize * kernelSize; ++i)
	{
		kernel[i] /= sum;
	}

	return kernel;
}

__host__ cv::Mat converToBgra(const char* fileName)
{
	cv::Mat imageBGRA, source = cv::imread(fileName, cv::IMREAD_UNCHANGED);
	cv::cvtColor(source, imageBGRA, cv::COLOR_BGR2BGRA);
	source.release();
	return imageBGRA;
}

long main()
{
	const size_t kernelSize = 5;
	cv::Mat image = converToBgra("source.png");
	float* kernel = createKernel(kernelSize);
	cudaDeviceProp deviceProp;

	cudaGetDeviceProperties(&deviceProp, 0);
	printDeviceProperties(deviceProp);

	CPU(image, kernel, kernelSize);
	GPU(image, kernel, kernelSize);
	GPUShared(image, kernel, kernelSize);

	delete[] kernel;
	image.release();
	system("pause");
	return 0;
}
#endif
