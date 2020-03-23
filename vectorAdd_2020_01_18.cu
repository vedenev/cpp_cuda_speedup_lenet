

#include <iostream>

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <stdlib.h>




__global__ void convolutions_valid_mode(float* features_input, int features_input_size_x, int features_input_size_y, int features_input_n_channels, 
	float* features_output, int features_output_size_x, int features_output_size_y, int features_output_n_channels,
	float* weights, int weights_size_x, int weights_size_y,
	float* biases)
{
	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;
	
	
	unsigned int output_1d_index = features_output_size_y * features_output_size_x * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;
	features_output[output_1d_index] = 0.0;
	for (int index_input_channel = 0; index_input_channel < features_input_n_channels; index_input_channel++)
	{
		unsigned int weights_1d_index_offset = weights_size_y * weights_size_x * features_input_n_channels * index_output_channel +
			weights_size_y * weights_size_x * index_input_channel;
		
		for (int weights_index_y = 0; weights_index_y < weights_size_y; weights_index_y++) {
			for (int weights_index_x = 0; weights_index_x < weights_size_x; weights_index_x++) {
				unsigned int index_input_x = index_output_x + weights_index_x;
				unsigned int index_input_y = index_output_y + weights_index_y;
				unsigned int input_1d_index = features_input_size_y * features_input_size_x * index_input_channel +
					features_input_size_x * index_input_y + index_input_x;
				unsigned int weights_1d_index = weights_1d_index_offset + weights_size_x * weights_index_y + weights_index_x;
				features_output[output_1d_index] += features_input[input_1d_index] * weights[weights_1d_index];

			}
		}
	}
	features_output[output_1d_index] += biases[index_output_channel];
}

int main(void)
{
	
	FILE* file;

	float* test_features_input_1d;
	test_features_input_1d = (float*)malloc(sizeof(float) * 2352);
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_features_input_1d.bin", "rb");
	fread(test_features_input_1d, sizeof(float), 2352, file);
	fclose(file);

	float* test_weights_1d;
	test_weights_1d = (float*)malloc(sizeof(float) * 270);
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_weights_1d.bin", "rb");
	fread(test_weights_1d, sizeof(float), 270, file);
	fclose(file);

	float* test_beases_1d;
	test_beases_1d = (float*)malloc(sizeof(float) * 10);
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_beases_1d.bin", "rb");
	fread(test_beases_1d, sizeof(float), 10, file);
	fclose(file);

	float* test_features_output_1d;
	test_features_output_1d = (float*)malloc(sizeof(float) * 6760);
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_features_output_1d.bin", "rb");
	fread(test_features_output_1d, sizeof(float), 6760, file);
	fclose(file);


	float* device_test_features_input_1d;
	cudaMalloc((void**)&device_test_features_input_1d, sizeof(float) * 2352);
	cudaMemcpy(device_test_features_input_1d, test_features_input_1d, sizeof(float) * 2352, cudaMemcpyHostToDevice);

	float* device_test_weights_1d;
	cudaMalloc((void**)&device_test_weights_1d, sizeof(float) * 270);
	cudaMemcpy(device_test_weights_1d, test_weights_1d, sizeof(float) * 270, cudaMemcpyHostToDevice);

	float* device_test_beases_1d;
	cudaMalloc((void**)&device_test_beases_1d, sizeof(float) * 10);
	cudaMemcpy(device_test_beases_1d, test_beases_1d, sizeof(float) * 10, cudaMemcpyHostToDevice);

	float* device_test_features_output_1d;
	cudaMalloc((void**)&device_test_features_output_1d, sizeof(float) * 6760);

	float* from_gpu_test_features_output_1d;
	from_gpu_test_features_output_1d = (float*)malloc(sizeof(float) * 6760);


	const dim3 threadBlock(13, 13, 5);
	const dim3 grid(2, 2, 2);


	
	convolutions_valid_mode<<<grid, threadBlock>>> (device_test_features_input_1d, 28, 28, 3,
		device_test_features_output_1d, 26, 26, 10,
		device_test_weights_1d, 3, 3,
		device_test_beases_1d);
	
	cudaDeviceSynchronize();

	cudaMemcpy(from_gpu_test_features_output_1d, device_test_features_output_1d, sizeof(float) * 6760, cudaMemcpyDeviceToHost);

	float diff_tmp_abs_max = 0.0;
	for (int out_1d_count = 0; out_1d_count < 6760; out_1d_count++)
	{
		float diff_tmp = from_gpu_test_features_output_1d[out_1d_count] - test_features_output_1d[out_1d_count];
		float diff_tmp_abs = diff_tmp;
		if (diff_tmp_abs < 0.0) {
			diff_tmp_abs = -diff_tmp_abs;
		}
		if (diff_tmp_abs > diff_tmp_abs_max)
		{
			diff_tmp_abs_max = diff_tmp_abs;
		}
	}

	std::cout << "diff_tmp_abs_max = " << diff_tmp_abs_max << std::endl;
	

	free(test_features_input_1d);
	free(test_weights_1d);
	free(test_beases_1d);
	free(test_features_output_1d);
	free(from_gpu_test_features_output_1d);

	cudaFree(device_test_features_input_1d);
	cudaFree(device_test_weights_1d);
	cudaFree(device_test_beases_1d);
	cudaFree(device_test_features_output_1d);
	


}