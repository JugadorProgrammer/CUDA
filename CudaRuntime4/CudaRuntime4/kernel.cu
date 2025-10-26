#pragma once
#ifndef __INTELLISENSE_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>

// размер блока или размер подматрицы
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
__global__ void gaussianBlurKernelShared(uchar* image, uchar* output, int width, int height, float* kernel, int kernelSize)
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

	//// понять, почему тут всё падает
	int outputIndex = (y * width + x) * 4;
	output[outputIndex] = (uchar)(sumB);     // Blue
	output[outputIndex + 1] = (uchar)(sumG); // Green
	output[outputIndex + 2] = (uchar)(sumR); // Red
	output[outputIndex + 3] = (uchar)(sumA); // Alpha
}

__global__ void gaussianBlurKernel(uchar* image, uchar* output, int width, int height, float* kernel, int kernelSize)
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

	//// понять, почему тут всё падает
	int outputIndex = (y * width + x) * 4;
	output[outputIndex] = (uchar)(sumB);     // Blue
	output[outputIndex + 1] = (uchar)(sumG); // Green
	output[outputIndex + 2] = (uchar)(sumR); // Red
	output[outputIndex + 3] = (uchar)(sumA); // Alpha
}

int main()
{
	srand(time(NULL));
	uchar* devImage = NULL, * devResultImage = NULL;
	float* kernel = NULL, * devKernel = NULL;

	int width, height, channelsCount, kernelSize = 5, radius = 2;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaDeviceProp deviceProp;
	float sigma = 1.0, milliseconds = 0, sum = 0;
	kernel = new float[kernelSize * kernelSize];

	for (int y = -radius; y <= radius; y++)
	{
		for (int x = -radius; x <= radius; x++)
		{
			float value = exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel[(y + radius) * kernelSize + (x + radius)] = value;
			sum += value;
		}
	}

	// Нормализация
	for (int i = 0; i < kernelSize * kernelSize; i++)
	{
		kernel[i] /= sum;
	}

	cv::Mat imageBGRA, resultMat, mat = cv::imread("source.png", cv::IMREAD_UNCHANGED);
	// Проверка, что изображение загружено успешно
	if (mat.empty())
	{
		fprintf(stderr, "Не удалось загрузить изображение!");
		goto Finish;
	}
	cv::cvtColor(mat, imageBGRA, cv::COLOR_BGR2BGRA);
	channelsCount = imageBGRA.channels();
	width = imageBGRA.cols;
	height = imageBGRA.rows;

	const dim3 blockDim(TILE_SIZE, TILE_SIZE), 
		gridDim((size_t)std::ceil((double)width / (double)TILE_SIZE), (size_t)std::ceil((double)height / (double)TILE_SIZE));
	///////////////////////////////////////GPU/////////////////////////////////////////////////////
	cudaStatus = cudaEventCreate(&start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&start) failed!");

	cudaStatus = cudaEventCreate(&stop);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventCreate(&stop) failed!");

	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	CHECK_CUDA_ERROR(cudaStatus, "cudaGetDeviceProperties failed!");

	printDeviceProperties(deviceProp);
	cudaStatus = cudaMalloc(&devImage, width * height * channelsCount * sizeof(uchar));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devImage failed!");

	cudaStatus = cudaMalloc(&devResultImage, width * height * channelsCount * sizeof(uchar));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devResultImage failed!");

	cudaStatus = cudaMalloc(&devKernel, kernelSize * kernelSize * sizeof(float));
	CHECK_CUDA_ERROR(cudaStatus, "cudaMalloc(&devKernel failed!");

	cudaStatus = cudaMemcpy(devKernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devKernel failed!");

	cudaStatus = cudaMemcpy(devImage, imageBGRA.data, width * height * channelsCount * sizeof(uchar), cudaMemcpyHostToDevice);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devImage failed!");

	printf("GPU start calculation\n");
	cudaStatus = cudaEventRecord(start);
	CHECK_CUDA_ERROR(cudaStatus, "cudaEventRecord(&start) failed!");

	//gaussianBlurKernel << <gridDim, blockDim >> > (devImage, devResultImage, width, height, devKernel, kernelSize);
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

	resultMat = cv::Mat(imageBGRA.rows, imageBGRA.cols, imageBGRA.type());
	cudaStatus = cudaMemcpy(resultMat.data, devResultImage, width * height * channelsCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	CHECK_CUDA_ERROR(cudaStatus, "cudaMemcpy(devResultImage failed!");

	printf("GPU time: %f ms\n", milliseconds);

	cv::imwrite("result.png", resultMat);
Finish:
	DELETE_ARRAY_IF_EXISTS(kernel);

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

	if (devKernel)
	{
		cudaStatus = cudaFree(devKernel);
		PRINT_CUDA_ERROR(cudaStatus, "cudaFree(devKernel failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	PRINT_CUDA_ERROR(cudaStatus, "cudaDeviceReset failed!");

	system("pause");
	return 0;
}
#endif
