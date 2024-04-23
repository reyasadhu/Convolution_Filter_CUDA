#include <torch/torch.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <opencv2/opencv.hpp>
#pragma comment(linker, "/INCLUDE:?ignore_this_library_placeholder@@YAHXZ")

using namespace std;
using namespace cv;

#define H_input 512
#define W_input 512
#define BLOCK_SIZE 16

#define MASK_WIDTH 3
#define RADIUS MASK_WIDTH/2



__global__ void global_sharpenFilter(float* d_Data, float* d_Result) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sharpenKernel[3][3] = { -1, -1, -1, -1,  9, -1, -1, -1, -1 };


	if ((x >= 1) && (x < (W_input - 1)) && (y >= 1) && (y < (H_input - 1)))
	{

		float sum = 0;


		for (int i = -RADIUS; i <= RADIUS; i++) {
			for (int j = -RADIUS; j <= RADIUS; j++) {

				float fl = d_Data[(y + i) * W_input + (x + j)];
				sum += fl * sharpenKernel[i + 1][j + 1];

			}
		}

		d_Result[y * W_input + x] = sum;
	}
}

__global__ void reorganized_shared_sharpenFilter(float* d_Data, float* d_Result)
{
	const int shm_width = BLOCK_SIZE + 2 * RADIUS;
	__shared__ float data[shm_width * shm_width];
	float sharpenKernel[9] = { -1, -1, -1, -1,  9, -1, -1, -1, -1 };

	int x, y; // Image-based coordinates

	// Original image-based coordinates
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	// global mem address of this thread
	const int gLoc = x0 + W_input * y0;

	// Case 1: upper left
	x = x0 - RADIUS;
	y = y0 - RADIUS;
	if (x < 0 || y < 0)
		data[threadIdx.y * shm_width + threadIdx.x] = 0;
	else
		data[threadIdx.y * shm_width + threadIdx.x] = d_Data[x + W_input * y];

	// Case 2: upper right
	x = x0 + RADIUS;
	y = y0 - RADIUS;
	if (x > W_input - 1 || y < 0)
		data[threadIdx.y * shm_width + threadIdx.x + 2 * RADIUS] = 0;
	else
		data[threadIdx.y * shm_width + threadIdx.x + 2 * RADIUS] = d_Data[x + W_input * y];

	// Case 3: lower left
	x = x0 - RADIUS;
	y = y0 + RADIUS;
	if (x < 0 || y > H_input - 1)
		data[(threadIdx.y + 2 * RADIUS) * shm_width + threadIdx.x] = 0;
	else
		data[(threadIdx.y + 2 * RADIUS) * shm_width + threadIdx.x] = d_Data[x + W_input * y];

	// Case 4: lower right
	x = x0 + RADIUS;
	y = y0 + RADIUS;
	if (x > W_input - 1 || y > H_input - 1)
		data[(threadIdx.y + 2 * RADIUS) * shm_width + threadIdx.x + 2 * RADIUS] = 0;
	else
		data[(threadIdx.y + 2 * RADIUS) * shm_width + threadIdx.x + 2 * RADIUS] = d_Data[x + W_input * y];

	__syncthreads();

	// Convolution
	float sum=0.0;

	for (int i = 0; i < MASK_WIDTH; i++) {
		for (int j = 0; j < MASK_WIDTH; j++) {
			sum += data[(threadIdx.y + i) * shm_width + threadIdx.x + j] * sharpenKernel[i * MASK_WIDTH + j];	
		}
	}


	d_Result[gLoc] = sum;
}


__global__ void shared_sobelEdgeDetection(float* d_Data, float* d_Result)
{
	float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };


	// Load cache (32x32 shared memory, 16x16 threads blocks)
	// Each thread loads four values from global memory into shared mem
	// If in image area, get value in global mem, else 0
	int x, y; // Image-based coordinates

	// Original image-based coordinates
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	// global mem address of this thread
	const int gLoc = x0 + W_input * y0;
	__shared__ float data[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];
	// Case 1: upper left
	x = x0 - RADIUS;
	y = y0 - RADIUS;
	if (x < 0 || y < 0)
		data[threadIdx.y][threadIdx.x] = 0;
	else
		data[threadIdx.y][threadIdx.x] = d_Data[x + W_input * y];

	// Case 2: upper right
	x = x0 + RADIUS;
	y = y0 - RADIUS;
	if (x > W_input - 1 || y < 0)
		data[threadIdx.y][threadIdx.x+2*RADIUS] = 0;
	else
		data[threadIdx.y][threadIdx.x+2*RADIUS] = d_Data[x+ W_input *y];

	// Case 3: lower left
	x = x0 - RADIUS;
	y = y0 + RADIUS;
	if (x < 0 || y > H_input - 1)
		data[threadIdx.y + 2 * RADIUS][threadIdx.x] = 0;
	else
		data[threadIdx.y + 2 * RADIUS][threadIdx.x] = d_Data[x + W_input * y];

	// Case 4: lower right
	x = x0 + RADIUS;
	y = y0 + RADIUS;
	if (x > W_input - 1 || y > H_input - 1)
		data[threadIdx.y +2 * RADIUS][threadIdx.x + 2 * RADIUS] = 0;
	else
		data[threadIdx.y + 2 * RADIUS][threadIdx.x +2 * RADIUS] = d_Data[x + W_input * y];

	__syncthreads();

	float Gx = 0,Gy=0;
	for (int i = 0; i <MASK_WIDTH; i++) {
		for (int j = 0; j <MASK_WIDTH; j++) {
			Gx += data[threadIdx.y + i][threadIdx.x + j] *Kx[i][j] ;
			Gy += data[threadIdx.y + i][threadIdx.x + j] *Ky[i][j] ;
		}
	}
	d_Result[gLoc] = sqrt(Gx*Gx+Gy*Gy);
}


__global__ void naive_shared_sobelEdgeDetection(float* d_Data, float* d_Result) 
	{
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		

		float Ky[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
		float Kx[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

		__shared__ float shm[BLOCK_SIZE][BLOCK_SIZE];
		shm[threadIdx.y][threadIdx.x] = d_Data[row * W_input + col];
		
		__syncthreads();
		float Gx = 0.0, Gy = 0.0;
		for (int i = 0; i < MASK_WIDTH; i++) {
			for (int j = 0; j < MASK_WIDTH; j++) {
				int current_x_global = col - RADIUS + j;
				int current_y_global = row - RADIUS + i;
				int current_x = threadIdx.x - RADIUS + j;
				int current_y = threadIdx.y - RADIUS + i;
				if (current_x_global>=0 && current_x_global<W_input && current_y_global >= 0 && current_y_global < H_input) {
					if (current_x >= 0 && current_x < BLOCK_SIZE && current_y >= 0 && current_y < BLOCK_SIZE) {
						Gx += shm[current_y][current_x] * Kx[i][j];
						Gy += shm[current_y][current_x] * Ky[i][j];
					}
					else {
						Gx += d_Data[current_y_global * W_input + current_x_global] * Kx[i][j];
						Gy += d_Data[current_y_global * W_input + current_x_global] * Ky[i][j];
					}
				}
			}
		}
			
		d_Result[row * W_input + col] = sqrt(Gx*Gx+Gy*Gy);
	
}

__global__ void reorganized_shared_sobelEdgeDetection(float* d_Data, float* d_Result) {
	
	const int shm_width = BLOCK_SIZE + 2 * RADIUS;
	__shared__ float data[shm_width*shm_width];
	
	float Kx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float Ky[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	int x, y; // Image-based coordinates

	// Original image-based coordinates
	const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
	const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
	// global mem address of this thread
	const int gLoc = x0 + W_input * y0;

	// Case 1: upper left
	x = x0 - RADIUS;
	y = y0 - RADIUS;
	if (x < 0 || y < 0)
		data[threadIdx.y*shm_width+threadIdx.x] = 0;
	else
		data[threadIdx.y * shm_width + threadIdx.x] = d_Data[x + W_input * y];

	// Case 2: upper right
	x = x0 + RADIUS;
	y = y0 - RADIUS;
	if (x > W_input - 1 || y < 0)
		data[threadIdx.y * shm_width + threadIdx.x + 2 * RADIUS] = 0;
	else
		data[threadIdx.y * shm_width + threadIdx.x + 2 * RADIUS] = d_Data[x + W_input * y];

	// Case 3: lower left
	x = x0 - RADIUS;
	y = y0 + RADIUS;
	if (x < 0 || y > H_input - 1)
		data[(threadIdx.y + 2 * RADIUS)* shm_width + threadIdx.x] = 0;
	else
		data[(threadIdx.y + 2 * RADIUS)* shm_width + threadIdx.x] = d_Data[x + W_input * y];

	// Case 4: lower right
	x = x0 + RADIUS;
	y = y0 + RADIUS;
	if (x > W_input - 1 || y > H_input - 1)
		data[(threadIdx.y + 2 * RADIUS)* shm_width + threadIdx.x + 2 * RADIUS] = 0;
	else
		data[(threadIdx.y + 2 * RADIUS) * shm_width + threadIdx.x + 2 * RADIUS] = d_Data[x + W_input * y];

	__syncthreads();

	// Convolution
	float Gx = 0, Gy = 0;

	for (int i = 0; i < MASK_WIDTH; i++) {
		for (int j = 0; j < MASK_WIDTH; j++) {
			Gx += data[(threadIdx.y + i)* shm_width + threadIdx.x + j] * Kx[i*MASK_WIDTH+j];
			Gy += data[(threadIdx.y + i)* shm_width + threadIdx.x + j] * Ky[i*MASK_WIDTH+j];
		}
	}
	d_Result[gLoc] = sqrt(Gx * Gx + Gy * Gy);
	
}



__global__ void global_sobelEdgeDetection(float* srcImage, float* dstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float Ky[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float Kx[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };


	// only threads inside image will write results
	if ((x >= 1) && (x < (W_input - 1)) && (y >= 1) && (y < (H_input - 1)))
	{

		float Gx = 0;
		float Gy = 0;

		for (int ky = -1; ky <= 1; ky++) {
			for (int kx = -1; kx <= 1; kx++) {

				float fl = srcImage[(y + ky) * W_input + (x + kx)];
				Gx += fl * Kx[ky + 1][kx + 1];
				Gy += fl * Ky[ky + 1][kx + 1];
			}
		}



		dstImage[y * W_input + x] = sqrt(Gx * Gx + Gy * Gy);
	}
}



torch::Device select_GPU(int id)
{
	torch::Device device(torch::kCPU);

	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		cudaDeviceProp cuda_dev;
		for (int w = 0; w < torch::cuda::device_count(); w++) {
			cudaGetDeviceProperties(&cuda_dev, w);
			printf("device id = %d, device name = %s\n", w, cuda_dev.name);
		}

		device = torch::Device(torch::kCUDA, id);
		cudaGetDeviceProperties(&cuda_dev, id);
		printf("use device id = %d, device name = %s\n\n", id, cuda_dev.name);
	}
	else {
		std::cout << "Training on CPU." << std::endl;
	}

	return device;
}


int main() {
	torch::Device device = select_GPU(0);
	std::string basePath = "H:\\Project\\src\\test\\";
	
	cv::Mat input = cv::imread((basePath+"input_image.png").c_str(), cv::IMREAD_GRAYSCALE);
	std::cout << "Check image dimensions" << std::endl;
	std::cout << "pixel dim x: " << input.rows << std::endl;
	std::cout << "pixel dim x: " << input.cols << std::endl;

	torch::Tensor tensor_image = torch::from_blob(input.data, { 1, input.rows, input.cols, 3 }, torch::kByte);

	tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	torch::Tensor tensor_image_GPU = tensor_image.to(device);


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((W_input + block.x - 1) / block.x, (H_input + block.y - 1) / block.y);
	char arr[6][50] = { "Edge:Global Memory","Edge: Naive Shared Memory","Edge:Shared Memory","Edge:Reorganized Shared Memory","Sharpen: Global Memory","Sharpen:Reorganized Shared Memory"};
	for (int i = 0; i < 6; i++) {
		torch::Tensor output_tensor = torch::empty_like(tensor_image);
		output_tensor = output_tensor.to(device);
		cudaEventRecord(start, 0);
		
		if (i == 0) {
			global_sobelEdgeDetection << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
		if (i == 1) {
			naive_shared_sobelEdgeDetection << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
		if (i == 2) {
			shared_sobelEdgeDetection << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
		if (i == 3) {
			reorganized_shared_sobelEdgeDetection << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
		if (i == 4) {
			global_sharpenFilter << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
		if (i == 5) {
			reorganized_shared_sharpenFilter << < grid, block >> > (tensor_image_GPU.data_ptr<float>(), output_tensor.data_ptr<float>());
		}
	

		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("\nProcessing time %s (ms): %f\n", arr[i], milliseconds);

		
		torch:: Tensor output_tensor_cpu = output_tensor.cpu();
		cv::Mat output_image(H_input, W_input, CV_32FC1, output_tensor_cpu.data_ptr<float>());
		if (i < 4) {
			cv::normalize(output_image, output_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		}
		std::ostringstream file_path_stream;
		file_path_stream << "D:\\projects\\Project\\src\\test\\results\\" << arr[i] << ".bmp";
		std::string file_path = file_path_stream.str();
		cv::imshow(arr[i], output_image);
		//cv::waitKey(0);
		cv::imwrite(file_path, output_image);
		

	}

	return 0;
}