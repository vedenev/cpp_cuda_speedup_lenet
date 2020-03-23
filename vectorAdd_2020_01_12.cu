

#include <iostream>

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
/**
 * CUDA Kernel Device code
 */



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
	//float sum_tmp = 0.0;
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
				//sum_tmp += features_input[input_1d_index] * weights[weights_1d_index];
				//test_weights[0, 1, 2, 2] = -0.74023384
				//test_weights[1, 1, 2, 2] = 1.6902944
				//if (index_output_x == 0 && index_output_y == 0 && weights_index_x == 2 && weights_index_y == 2 && index_input_channel == 1 && features_output_index_to_calculate == 1)
				//{
				//	printf("test_weights:");
				//	printf("weights_1d_index =%d\n", weights_1d_index);
				//	printf("weights[weights_1d_index] =%.6f\n", weights[weights_1d_index]);
				//}
			}
		}
	}
	features_output[output_1d_index] += biases[index_output_channel];
	//sum_tmp += biases[index_output_channel];
	//features_output[output_1d_index] = sum_tmp;
	
	//printf("%d\n", index_output_channel);
	
	
	//biases[index_output_channel] = 1000000.0; // for debug
	//printf("%d %d %d %.6f\n", index_output_x, index_output_y, index_output_channel, biases[index_output_channel]);
}

int main(void)
{
	/*
	float a[5];
	FILE * file = fopen("F:/freelance/cpp_learning/cuda_learning/a.bin", "rb");
	fread(a, sizeof(float), 5, file);
	fclose(file);
	for (int count = 0; count < 5; count++)
	{
		std::cout << a[count] << std::endl;
	}
	*/

	FILE* file;

	float test_features_input_1d[2352];
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_features_input_1d.bin", "rb");
	fread(test_features_input_1d, sizeof(float), 2352, file);
	fclose(file);

	float test_weights_1d[270];
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_weights_1d.bin", "rb");
	fread(test_weights_1d, sizeof(float), 270, file);
	fclose(file);

	float test_beases_1d[10];
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_beases_1d.bin", "rb");
	fread(test_beases_1d, sizeof(float), 10, file);
	fclose(file);

	float test_features_output_1d[6760];
	file = fopen("F:/freelance/cpp_learning/cuda_learning/test_features_output_1d.bin", "rb");
	fread(test_features_output_1d, sizeof(float), 6760, file);
	fclose(file);


	float* device_test_features_input_1d;
	cudaMalloc((void**)&device_test_features_input_1d, sizeof(float) * 2352);
	cudaMemcpy(device_test_features_input_1d, test_features_input_1d, sizeof(float) * 2352, cudaMemcpyHostToDevice);

	float* device_test_weights_1d;
	cudaMalloc((void**)&device_test_weights_1d, sizeof(float) * 270);
	cudaMemcpy(device_test_weights_1d, test_weights_1d, sizeof(float) * 270, cudaMemcpyHostToDevice);
	//printf("test_weights_1d[17] = %.6f", test_weights_1d[17]);

	float* device_test_beases_1d;
	cudaMalloc((void**)&device_test_beases_1d, sizeof(float) * 10);
	cudaMemcpy(device_test_beases_1d, test_beases_1d, sizeof(float) * 10, cudaMemcpyHostToDevice);

	float* device_test_features_output_1d;
	cudaMalloc((void**)&device_test_features_output_1d, sizeof(float) * 6760);

	float from_gpu_test_features_output_1d[6760];

	//float tmp[1];
	//tmp[0] = 0.0;


	//float* device_tmp;
	//cudaMalloc((void**)&device_tmp, sizeof(float) * 1);
	//cudaMemcpy(device_tmp, tmp, sizeof(float) * 1, cudaMemcpyHostToDevice);

	const dim3 threadBlock(13, 13, 5);
	const dim3 grid(2, 2, 2);
	   
	//const dim3 threadBlock(26, 26, 1);
	//const dim3 grid(1, 1, 10);

	//const dim3 threadBlock(13, 13, 1);
	//const dim3 grid(2, 2, 10);

	//const dim3 threadBlock(1, 1, 1);
	//const dim3 grid(1, 1, 10);

	//const dim3 threadBlock(1, 1, 10);
	//const dim3 grid(1, 1, 1);

	
	convolutions_valid_mode<<<grid, threadBlock>>> (device_test_features_input_1d, 28, 28, 3,
		device_test_features_output_1d, 26, 26, 10,
		device_test_weights_1d, 3, 3,
		device_test_beases_1d);
	
	cudaDeviceSynchronize();

	cudaMemcpy(from_gpu_test_features_output_1d, device_test_features_output_1d, sizeof(float) * 6760, cudaMemcpyDeviceToHost);

	float diff_tmp_abs_max = 0.0;
	for (int out_1d_count = 0; out_1d_count < 6760; out_1d_count++)
	//for (int out_1d_count = 0; out_1d_count < 1000; out_1d_count++)
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

		if (diff_tmp_abs_max > 1e-4)
		{
			std::cout << "out_1d_count = " << out_1d_count << std::endl;
			std::cout << "diff_tmp_abs_max = " << diff_tmp_abs_max << std::endl;
			break;
		}
	}

	std::cout << "diff_tmp_abs_max = " << diff_tmp_abs_max << std::endl;
	
	/*
	cudaDeviceSynchronize();
	cudaMemcpy(test_beases_1d, device_test_beases_1d, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for (int count = 0; count < 10; count++)
	{
		std::cout << test_beases_1d[count] << std::endl;
	}
	*/


}