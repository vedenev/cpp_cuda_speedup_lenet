

#include <iostream>

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <helper_cuda.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>





//const unsigned int W_conv1_size_const = 144;
//__device__ __constant__ float W_conv1_const[W_conv1_size_const];
//__constant__ float W_conv1_const[W_conv1_size_const];
const unsigned int constants_size = (144 + 16 + 6400 + 16);
__constant__ float constants[constants_size];








__global__ void convolutions_relu_constants_weights_shared_memory(int input_offset, float* features_input, int features_input_size_x, int features_input_step, int features_input_n_channels, int features_input_size,
	float* features_output, int features_output_size_x, int features_output_step, int features_output_n_channels,
	int weights_offset, int weights_size_x, int weights_size_y, int weights_step_1, int weights_step_2,
	int biases_offset,
	int n_threds_in_block, int threds_step_y_x
	)
{

	//extern __shared__ float features_input_cache[];
	__shared__ float features_input_cache[2704]; //, 2688 - not ok, 2689 - ok, 13*13*16 = 2704

	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;

	// fill features_input_cache:
	//unsigned int features_output_step = features_output_size_y * features_output_size_x;
	//unsigned int features_input_step = features_input_size_y * features_input_size_x;
	//unsigned int features_input_size = features_input_step * features_input_n_channels;
	unsigned int output_1d_index = features_output_step * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;
	
	//unsigned int n_threds_in_block = blockDim.x * blockDim.y * blockDim.z;
	unsigned int threds_1d_index = threadIdx.z * threds_step_y_x + threadIdx.y * blockDim.x + threadIdx.x;

	
	if (features_input_size == n_threds_in_block)
	{
		features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
	}
	else if (features_input_size < n_threds_in_block)
	{
		if (threds_1d_index < features_input_size)
		{
			features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
		}
	}
	else
	{
		
		// case for features_input_size > n_threds_in_block
		unsigned int index_1 = (threds_1d_index * features_input_size) / n_threds_in_block;
		unsigned int index_2 = ((threds_1d_index + 1) * features_input_size) / n_threds_in_block;
		for (unsigned int index = index_1; index < index_2; index++)
		{
			features_input_cache[index] = features_input[input_offset + index];
		}
		
	}
	



	__syncthreads();
	

	// convolutions:
	

	//unsigned int weights_step_1 = weights_size_y * weights_size_x;
	//unsigned int weights_step_2 = weights_step_1 * features_input_n_channels * index_output_channel;

	unsigned int weights_step_2_2 = weights_step_2 * index_output_channel;

	



	float output_value = 0.0;

	for (int index_input_channel = 0; index_input_channel < features_input_n_channels; index_input_channel++)
	{
		unsigned int weights_1d_index_offset = weights_step_2_2 + weights_step_1 * index_input_channel;

		for (int weights_index_y = 0; weights_index_y < weights_size_y; weights_index_y++) {
			for (int weights_index_x = 0; weights_index_x < weights_size_x; weights_index_x++) {
				unsigned int index_input_x = index_output_x + weights_index_x;
				unsigned int index_input_y = index_output_y + weights_index_y;
				unsigned int input_1d_index = features_input_step * index_input_channel +
					features_input_size_x * index_input_y + index_input_x;
				unsigned int weights_1d_index = weights_1d_index_offset + weights_size_x * weights_index_y + weights_index_x;
				output_value += features_input_cache[input_1d_index] * constants[weights_offset + weights_1d_index];
				//output_value += features_input[input_offset + input_1d_index] * constants[weights_offset + weights_1d_index];
				//output_value += features_input_cache[2703] * constants[weights_offset + weights_1d_index];
				

			}
		}

	}

	output_value += constants[biases_offset + index_output_channel];


	output_value = fmaxf(output_value, 0.0); // relu

	features_output[output_1d_index] = output_value;

}


__global__ void convolutions_relu_shared_memory(int input_offset, float* features_input, int features_input_size_x, int features_input_step, int features_input_n_channels, int features_input_size,
	float* features_output, int features_output_size_x, int features_output_size_y_x, int features_output_n_channels,
	float* weights, int weights_size_x, int weights_size_y, int weights_step_1, int weights_step_2,
	float* biases, 
	int n_threds_in_block, int threds_step_y_x)
{

	__shared__ float features_input_cache[144];

	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;


	//unsigned int n_threds_in_block = blockDim.x * blockDim.y * blockDim.z;
	unsigned int threds_1d_index = threadIdx.z * threds_step_y_x + threadIdx.y * blockDim.x + threadIdx.x;

	//unsigned int features_input_step = features_input_size_y * features_input_size_x;
	//unsigned int features_input_size = features_input_step * features_input_n_channels;

	if (features_input_size == n_threds_in_block)
	{
		features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
	}
	else if (features_input_size < n_threds_in_block)
	{
		if (threds_1d_index < features_input_size)
		{
			features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
		}
	}
	else
	{

		// case for features_input_size > n_threds_in_block
		unsigned int index_1 = (threds_1d_index * features_input_size) / n_threds_in_block;
		unsigned int index_2 = ((threds_1d_index + 1) * features_input_size) / n_threds_in_block;
		for (unsigned int index = index_1; index < index_2; index++)
		{
			features_input_cache[index] = features_input[input_offset + index];
		}

	}




	__syncthreads();

	//unsigned int features_output_size_y_x = features_output_size_y * features_output_size_x;
	unsigned int output_1d_index = features_output_size_y_x * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;

	//unsigned int weights_step_1 = weights_size_y * weights_size_x;
	//unsigned int weights_step_2 = weights_step_1 * features_input_n_channels * index_output_channel;
	unsigned int weights_step_2_2 = weights_step_2 * index_output_channel;

	float output_value = 0.0;

	for (int index_input_channel = 0; index_input_channel < features_input_n_channels; index_input_channel++)
	{
		unsigned int weights_1d_index_offset = weights_step_2_2 + weights_step_1 * index_input_channel;

		for (int weights_index_y = 0; weights_index_y < weights_size_y; weights_index_y++) {
			for (int weights_index_x = 0; weights_index_x < weights_size_x; weights_index_x++) {
				unsigned int index_input_x = index_output_x + weights_index_x;
				unsigned int index_input_y = index_output_y + weights_index_y;
				unsigned int input_1d_index = features_input_step * index_input_channel +
					features_input_size_x * index_input_y + index_input_x;
				unsigned int weights_1d_index = weights_1d_index_offset + weights_size_x * weights_index_y + weights_index_x;
				output_value += features_input_cache[input_1d_index] * weights[weights_1d_index];

			}
		}

	}

	output_value += biases[index_output_channel];


	output_value = fmaxf(output_value, 0.0); // relu

	features_output[output_1d_index] = output_value;

}


__global__ void convolutions_shared_memory(int input_offset, float* features_input, int features_input_size_x, int features_input_step, int features_input_n_channels, int features_input_size,
	float* features_output, int features_output_size_x, int features_output_size_y_x, int features_output_n_channels,
	float* weights, int weights_size_x, int weights_size_y, int weights_step_1, int weights_step_2,
	float* biases,
	int n_threds_in_block, int threds_step_y_x)
{

	__shared__ float features_input_cache[256];

	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;


	//unsigned int n_threds_in_block = blockDim.x * blockDim.y * blockDim.z;
	unsigned int threds_1d_index = threadIdx.z * threds_step_y_x + threadIdx.y * blockDim.x + threadIdx.x;

	//unsigned int features_input_step = features_input_size_y * features_input_size_x;
	//unsigned int features_input_size = features_input_step * features_input_n_channels;

	if (features_input_size == n_threds_in_block)
	{
		features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
	}
	else if (features_input_size < n_threds_in_block)
	{
		if (threds_1d_index < features_input_size)
		{
			features_input_cache[threds_1d_index] = features_input[input_offset + threds_1d_index];
		}
	}
	else
	{

		// case for features_input_size > n_threds_in_block
		unsigned int index_1 = (threds_1d_index * features_input_size) / n_threds_in_block;
		unsigned int index_2 = ((threds_1d_index + 1) * features_input_size) / n_threds_in_block;
		for (unsigned int index = index_1; index < index_2; index++)
		{
			features_input_cache[index] = features_input[input_offset + index];
		}

	}




	__syncthreads();

	//unsigned int features_output_size_y_x = features_output_size_y * features_output_size_x;
	unsigned int output_1d_index = features_output_size_y_x * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;

	//unsigned int weights_step_1 = weights_size_y * weights_size_x;
	//unsigned int weights_step_2 = weights_step_1 * features_input_n_channels * index_output_channel;
	unsigned int weights_step_2_2 = weights_step_2 * index_output_channel;

	float output_value = 0.0;

	for (int index_input_channel = 0; index_input_channel < features_input_n_channels; index_input_channel++)
	{
		unsigned int weights_1d_index_offset = weights_step_2_2 + weights_step_1 * index_input_channel;

		for (int weights_index_y = 0; weights_index_y < weights_size_y; weights_index_y++) {
			for (int weights_index_x = 0; weights_index_x < weights_size_x; weights_index_x++) {
				unsigned int index_input_x = index_output_x + weights_index_x;
				unsigned int index_input_y = index_output_y + weights_index_y;
				unsigned int input_1d_index = features_input_step * index_input_channel +
					features_input_size_x * index_input_y + index_input_x;
				unsigned int weights_1d_index = weights_1d_index_offset + weights_size_x * weights_index_y + weights_index_x;
				output_value += features_input_cache[input_1d_index] * weights[weights_1d_index];

			}
		}

	}

	output_value += biases[index_output_channel];

	features_output[output_1d_index] = output_value;

}

__global__ void max_pooling_2x2(float* features_input, int features_input_size_x, int features_input_size_y_x, int features_input_n_channels, 
	float* features_output, int features_output_size_x, int features_output_size_y_x)
{
	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;

	unsigned int index_input_x = 2 * index_output_x;
	unsigned int index_input_y = 2 * index_output_y;

	unsigned int output_1d_index = features_output_size_y_x * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;

	unsigned int features_input_step = features_input_size_y_x * index_output_channel;
	unsigned int input_1d_index_0_0 = features_input_step +
		features_input_size_x * index_input_y + index_input_x;
	unsigned int input_1d_index_0_1 = input_1d_index_0_0 + 1;
	unsigned int input_1d_index_1_0 = input_1d_index_0_0 + features_input_size_x;
	unsigned int input_1d_index_1_1 = input_1d_index_0_0 + 1 + features_input_size_x;

	float max_0 = fmaxf(features_input[input_1d_index_0_0], features_input[input_1d_index_0_1]);
	float max_1 = fmaxf(features_input[input_1d_index_1_0], features_input[input_1d_index_1_1]);
	features_output[output_1d_index] = fmaxf(max_0, max_1);
}

__global__ void max_pooling_3x3(float* features_input, int features_input_size_x, int features_input_size_y_x, int features_input_n_channels,
	float* features_output, int features_output_size_x, int features_output_size_y_x)
{
	const unsigned int index_output_x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int index_output_y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int index_output_channel = blockIdx.z * blockDim.z + threadIdx.z;

	unsigned int index_input_x = 3 * index_output_x;
	unsigned int index_input_y = 3 * index_output_y;

	unsigned int output_1d_index = features_output_size_y_x * index_output_channel +
		features_output_size_x * index_output_y + index_output_x;

	unsigned int features_input_step = features_input_size_y_x * index_output_channel;
	unsigned int input_1d_index_0_0 = features_input_step +
		features_input_size_x * index_input_y + index_input_x;
	unsigned int input_1d_index_0_1 = input_1d_index_0_0 + 1;
	unsigned int input_1d_index_0_2 = input_1d_index_0_0 + 2;
	unsigned int input_1d_index_1_0 = input_1d_index_0_0 + features_input_size_x;
	unsigned int input_1d_index_1_1 = input_1d_index_1_0 + 1;
	unsigned int input_1d_index_1_2 = input_1d_index_1_0 + 2;
	unsigned int input_1d_index_2_0 = input_1d_index_1_0 + features_input_size_x;
	unsigned int input_1d_index_2_1 = input_1d_index_2_0 + 1;
	unsigned int input_1d_index_2_2 = input_1d_index_2_0 + 2;

	float max_0 = fmaxf(features_input[input_1d_index_0_0], features_input[input_1d_index_0_1]);
	float max_1 = fmaxf(features_input[input_1d_index_0_2], features_input[input_1d_index_1_0]);
	float max_2 = fmaxf(features_input[input_1d_index_1_1], features_input[input_1d_index_1_2]);
	float max_3 = fmaxf(features_input[input_1d_index_2_0], features_input[input_1d_index_2_1]);

	float max_4 = fmaxf(max_0, max_1);
	float max_5 = fmaxf(max_2, max_3);

	float max_6 = fmaxf(max_4, max_5);

	features_output[output_1d_index] = fmaxf(max_6, features_input[input_1d_index_2_2]);
}

void c_stings_concatinate(char* string_1, char* string_2, char** string_result)
{
	*string_result = (char*)malloc(strlen(string_1) + strlen(string_2) + 1);
	strcpy(*string_result, string_1);
	strcat(*string_result, string_2);

}

void load_data_to_array(char* dir, char* file, float** array_gpu, int size)
{
	// https://stackoverflow.com/questions/22826380/cuda-allocation-and-return-array-from-gpu-to-cpu
	char* path;
	c_stings_concatinate(dir, file, &path);
	float* array_cpu;
	array_cpu = (float*)malloc(sizeof(float) * size);
	FILE* file_id;
	file_id = fopen(path, "rb");
	int n_floats_readed = fread(array_cpu, sizeof(float), size, file_id);
	fclose(file_id);
	if (n_floats_readed != size)
	{
		printf("n_floats_readed != size   n_floats_readed = %d  size = %d\n", n_floats_readed, size);
	}
	if (cudaMalloc((void**)array_gpu, sizeof(float) * size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU\n";
	}
	
	cudaMemcpy(*array_gpu, array_cpu, sizeof(float) * size, cudaMemcpyHostToDevice);

	
	//float* array_cpu_check;
	//array_cpu_check = (float*)malloc(sizeof(float) * size);
	//cudaMemcpy(array_cpu_check, array_gpu, sizeof(float) * size, cudaMemcpyDeviceToHost);
	//for (int counter = size-1; counter < size; counter++)
	//{
	//	printf("array_cpu_check[counter] = %.6f    array_cpu[counter] = %.6f\n", array_cpu_check[counter], array_cpu[counter]);
	//}
	//free(array_cpu_check);
	

	free(array_cpu);
	free(path);
}

void append_data_to_cpu_array(char* dir, char* file, float* constants_cpu, int size, int* offset)
{
	char* path;
	c_stings_concatinate(dir, file, &path);
	float* pointer_shifted;
	FILE* file_id;
	file_id = fopen(path, "rb");
	pointer_shifted = constants_cpu + *offset;
	int n_floats_readed = fread(pointer_shifted, sizeof(float), size, file_id);
	fclose(file_id);
	*offset += size;
	free(path);

}

/*
__global__ void tmp_check(float* array_cpu_4, float* accuracy_ptr)
{
	accuracy_ptr[0] = 0.0;
	for (int weights_1d_index = 0; weights_1d_index < W_conv1_size_const; weights_1d_index++)
	{
		if (W_conv1_const[weights_1d_index] == array_cpu_4[weights_1d_index])
		//if (W_conv1_const[weights_1d_index] == 0.0)
		{
			accuracy_ptr[0] += 1.0;
		}
		//accuracy_ptr[0] += W_conv1_const[weights_1d_index];
	}
	accuracy_ptr[0] /= W_conv1_size_const;
}
*/

__global__ void check_constants(float* array_cpu, int size, float* n_correct_ptr)
{
	for (int index = 0; index < size; index++)
	{
		if (constants[index] == array_cpu[index])
		{
			n_correct_ptr[0] += 1.0;
		}
	}
}

int main(void)
{
	
	
	char* weights_dir = "F:/freelance/cpp_learning/cuda_learning/weigths_1d/";
	



	

	//              c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
	// 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
	// n mult : 97344                                 518400                          36864             2560
	// 784                   36864           2704               1296              144                256            10
	int input_size_x = 28;
	int input_size_y = 28;
	int input_n_channels = 1;
	int n_output = 10;

	int input_step = input_size_x * input_size_y;
	int input_size = input_size_x * input_size_y * input_n_channels;

	int featuremaps_1_size_x = 26;
	int featuremaps_1_size_y = 26;
	int featuremaps_1_size_y_x = featuremaps_1_size_x * featuremaps_1_size_y;
	int featuremaps_1_n_channels = 16;
	int featuremaps_1_size = featuremaps_1_size_x * featuremaps_1_size_y * featuremaps_1_n_channels;
	int featuremaps_1_thread_size_x = 26;
	int featuremaps_1_thread_size_y = 26;
	int featuremaps_1_thread_size_z = 1;
	int featuremaps_1_thread_size = featuremaps_1_thread_size_x * featuremaps_1_thread_size_y * featuremaps_1_thread_size_z;
	int featuremaps_1_threds_step_y_x = featuremaps_1_thread_size_x * featuremaps_1_thread_size_y;
	int featuremaps_1_greed_size_x = featuremaps_1_size_x / featuremaps_1_thread_size_x;
	int featuremaps_1_greed_size_y = featuremaps_1_size_y / featuremaps_1_thread_size_y;
	int featuremaps_1_greed_size_z = featuremaps_1_n_channels / featuremaps_1_thread_size_z;

	int featuremaps_1_pooling_size_x = 13;
	int featuremaps_1_pooling_size_y = 13;
	int featuremaps_1_pooling_n_channels = featuremaps_1_n_channels;
	int featuremaps_1_pooling_size_y_x = featuremaps_1_pooling_size_x * featuremaps_1_pooling_size_y;
	int featuremaps_1_pooling_size = featuremaps_1_pooling_size_x * featuremaps_1_pooling_size_y * featuremaps_1_n_channels;
	int featuremaps_1_pooling_thread_size_x = 13;
	int featuremaps_1_pooling_thread_size_y = 13;
	int featuremaps_1_pooling_thread_size_z = 4;
	int featuremaps_1_pooling_greed_size_x = featuremaps_1_pooling_size_x / featuremaps_1_pooling_thread_size_x;
	int featuremaps_1_pooling_greed_size_y = featuremaps_1_pooling_size_y / featuremaps_1_pooling_thread_size_y;
	int featuremaps_1_pooling_greed_size_z = featuremaps_1_n_channels / featuremaps_1_pooling_thread_size_z;

	int featuremaps_2_size_x = 9;
	int featuremaps_2_size_y = 9;
	int featuremaps_2_size_y_x = featuremaps_2_size_x * featuremaps_2_size_y;
	int featuremaps_2_n_channels = 16;
	int featuremaps_2_size = featuremaps_2_size_x * featuremaps_2_size_y * featuremaps_2_n_channels;
	int featuremaps_2_thread_size_x = 9;
	int featuremaps_2_thread_size_y = 9;
	int featuremaps_2_thread_size_z = 8;
	int featuremaps_2_thread_size = featuremaps_2_thread_size_x * featuremaps_2_thread_size_y * featuremaps_2_thread_size_z;
	int featuremaps_2_threds_step_y_x = featuremaps_2_thread_size_x * featuremaps_2_thread_size_y;
	int featuremaps_2_greed_size_x = featuremaps_2_size_x / featuremaps_2_thread_size_x;
	int featuremaps_2_greed_size_y = featuremaps_2_size_y / featuremaps_2_thread_size_y;
	int featuremaps_2_greed_size_z = featuremaps_2_n_channels / featuremaps_2_thread_size_z;

	int featuremaps_2_pooling_size_x = 3;
	int featuremaps_2_pooling_size_y = 3;
	int featuremaps_2_pooling_n_channels = featuremaps_2_n_channels;
	int featuremaps_2_pooling_size_y_x = featuremaps_2_pooling_size_x * featuremaps_2_pooling_size_y;
	int featuremaps_2_pooling_size = featuremaps_2_pooling_size_x * featuremaps_2_pooling_size_y * featuremaps_2_n_channels;
	int featuremaps_2_pooling_thread_size_x = 3;
	int featuremaps_2_pooling_thread_size_y = 3;
	int featuremaps_2_pooling_thread_size_z = 4;
	int featuremaps_2_pooling_greed_size_x = featuremaps_2_pooling_size_x / featuremaps_2_pooling_thread_size_x;
	int featuremaps_2_pooling_greed_size_y = featuremaps_2_pooling_size_y / featuremaps_2_pooling_thread_size_y;
	int featuremaps_2_pooling_greed_size_z = featuremaps_2_n_channels / featuremaps_2_pooling_thread_size_z;

	int featuremaps_3_size_x = 1;
	int featuremaps_3_size_y = 1;
	int featuremaps_3_n_channels = 256;
	int featuremaps_3_size = featuremaps_3_size_x * featuremaps_3_size_y * featuremaps_3_n_channels;
	int featuremaps_3_size_y_x = featuremaps_3_size_x * featuremaps_3_size_y;
	int featuremaps_3_thread_size_x = 1;
	int featuremaps_3_thread_size_y = 1;
	int featuremaps_3_thread_size_z = 64;
	int featuremaps_3_thread_size = featuremaps_3_thread_size_x * featuremaps_3_thread_size_y * featuremaps_3_thread_size_z;
	int featuremaps_3_threds_step_y_x = featuremaps_3_thread_size_x * featuremaps_3_thread_size_y;
	int featuremaps_3_greed_size_x = featuremaps_3_size_x / featuremaps_3_thread_size_x;
	int featuremaps_3_greed_size_y = featuremaps_3_size_y / featuremaps_3_thread_size_y;
	int featuremaps_3_greed_size_z = featuremaps_3_n_channels / featuremaps_3_thread_size_z;

	int featuremaps_4_size_x = 1;
	int featuremaps_4_size_y = 1;
	int featuremaps_4_n_channels = n_output;
	int featuremaps_4_size = featuremaps_4_size_x * featuremaps_4_size_y * featuremaps_4_n_channels;
	int featuremaps_4_size_y_x = featuremaps_4_size_x * featuremaps_4_size_y;
	int featuremaps_4_thread_size_x = 1;
	int featuremaps_4_thread_size_y = 1;
	int featuremaps_4_thread_size_z = 10;
	int featuremaps_4_thread_size = featuremaps_4_thread_size_x * featuremaps_4_thread_size_y * featuremaps_4_thread_size_z;
	int featuremaps_4_threds_step_y_x = featuremaps_4_thread_size_x * featuremaps_4_thread_size_y;
	int featuremaps_4_greed_size_x = featuremaps_4_size_x / featuremaps_4_thread_size_x;
	int featuremaps_4_greed_size_y = featuremaps_4_size_y / featuremaps_4_thread_size_y;
	int featuremaps_4_greed_size_z = featuremaps_4_n_channels / featuremaps_4_thread_size_z;

	int W_conv1_size_x = 3;
	int W_conv1_size_y = 3;
	int W_conv1_size = W_conv1_size_x * W_conv1_size_y * input_n_channels * featuremaps_1_n_channels;
	int W_conv1_step_1 = W_conv1_size_x * W_conv1_size_y;
	int W_conv1_step_2 = W_conv1_size_x * W_conv1_size_y * input_n_channels;
	int b_conv1_size = featuremaps_1_n_channels;

	int W_conv2_size_x = 5;
	int W_conv2_size_y = 5;
	int W_conv2_size = W_conv2_size_x * W_conv2_size_y * featuremaps_1_n_channels * featuremaps_2_n_channels;
	int W_conv2_step_1 = W_conv2_size_x * W_conv2_size_y;
	int W_conv2_step_2 = W_conv2_size_x * W_conv2_size_y * featuremaps_1_n_channels;
	int b_conv2_size = featuremaps_2_n_channels;

	int W_conv3_size_x = 3;
	int W_conv3_size_y = 3;
	int W_conv3_size = W_conv3_size_x * W_conv3_size_y * featuremaps_2_n_channels * featuremaps_3_n_channels;
	int W_conv3_step_1 = W_conv3_size_x * W_conv3_size_y;
	int W_conv3_step_2 = W_conv3_size_x * W_conv3_size_y * featuremaps_2_n_channels;
	int b_conv3_size = featuremaps_3_n_channels;

	int W_conv4_size_x = 1;
	int W_conv4_size_y = 1;
	int W_conv4_size = W_conv4_size_x * W_conv4_size_y * featuremaps_3_n_channels * featuremaps_4_n_channels;
	int W_conv4_step_1 = W_conv4_size_x * W_conv4_size_y;
	int W_conv4_step_2 = W_conv4_size_x * W_conv4_size_y * featuremaps_3_n_channels;
	int b_conv4_size = 10;

	int x_val_size = 7840000;
	int n_samples = 10000;
	
	//constants

	float* constants_cpu;
	constants_cpu = (float*)malloc(sizeof(float) * constants_size);
	int offset = 0;
	int offset_W_conv1 = offset;
	append_data_to_cpu_array(weights_dir, "W_conv1.bin", constants_cpu, W_conv1_size, &offset);
	int offset_b_conv1 = offset;
	append_data_to_cpu_array(weights_dir, "b_conv1.bin", constants_cpu, b_conv1_size, &offset);
	int offset_W_conv2 = offset;
	append_data_to_cpu_array(weights_dir, "W_conv2.bin", constants_cpu, W_conv2_size, &offset);
	int offset_b_conv2 = offset;
	append_data_to_cpu_array(weights_dir, "b_conv2.bin", constants_cpu, b_conv2_size, &offset);
	

	//for (int index = 0; index < constants_size; index++)
	//{
	//	printf("%.6f\n", constants_cpu[index]);
	//	
	//}

	checkCudaErrors(cudaMemcpyToSymbol(constants, constants_cpu, sizeof(float)* constants_size));
	float* n_correct_ptr;
	if (cudaMalloc((void**)&n_correct_ptr, sizeof(float) * 1) != cudaSuccess)
	{
		std::cout << "Error allocating GPU n_correct_ptr\n";
	}
	//check_constants<<<1, 1>>>(constants_cpu, constants_size, n_correct_ptr);
	//cudaDeviceSynchronize();
	float* n_correct_ptr_cpu;
	n_correct_ptr_cpu = (float*)malloc(sizeof(float) * 1);
	cudaMemcpy(n_correct_ptr_cpu, n_correct_ptr, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	printf("\n");
	printf("check constants:\n");
	printf("n_correct_ptr_cpu[0] = %.6f\n", n_correct_ptr_cpu[0]);
	printf("constants_size =%d\n", constants_size);
	printf("\n");
	float* constants_cpu_2;
	constants_cpu_2 = (float*)malloc(sizeof(float) * constants_size);
	checkCudaErrors(cudaMemcpyFromSymbol(constants_cpu_2, constants, sizeof(float) * constants_size));
	int is_equal = 0;
	int last_correct_index = 0;
	for (int index = 0; index < constants_size; index++)
	{
		if (constants_cpu_2[index] == constants_cpu[index])
		{
			is_equal = 1;
			last_correct_index = index;
		}
		else
		{
			is_equal = 0;
		}
		printf("%.6f   %.6f   %d\n", constants_cpu_2[index], constants_cpu[index], is_equal);
	}
	printf("last_correct_index = %d\n", last_correct_index);
	//last_correct_index = 6792
	cudaFree(n_correct_ptr);
	free(n_correct_ptr_cpu);
	


	float* W_conv3;
	float* b_conv3;
	float* W_conv4;
	float* b_conv4;
	float* x_val;
	
	//W_conv1_1d.size = 144
	//b_conv1_1d.size = 16
	//W_conv2_1d.size = 6400
	//b_conv2_1d.size = 16
	//W_conv3_1d.size = 36864
	//b_conv3_1d.size = 256
	//W_conv4_1d.size = 2560
	//b_conv4_1d.size = 10
	//x_val_1d.size = 7840000
	//y_val.size = 10000

	//(144 + 16 + 6400 + 16    +   256 + 2560 + 10)*4 = 37608
	
	load_data_to_array(weights_dir, "W_conv3.bin", &W_conv3, W_conv3_size);
	load_data_to_array(weights_dir, "b_conv3.bin", &b_conv3, b_conv3_size);
	load_data_to_array(weights_dir, "W_conv4.bin", &W_conv4, W_conv4_size);
	load_data_to_array(weights_dir, "b_conv4.bin", &b_conv4, b_conv4_size);
	load_data_to_array(weights_dir, "x_val.bin", &x_val, x_val_size);
	
	char* y_val_cpu;
	y_val_cpu = (char*)malloc(sizeof(char) * n_samples);
	FILE* file_id;
	char* path;
	c_stings_concatinate(weights_dir, "y_val.bin", &path);
	file_id = fopen(path, "rb");
	fread(y_val_cpu, sizeof(float), n_samples, file_id);
	fclose(file_id);
	free(path);

	float* featuremaps_1;
	if (cudaMalloc((void**)&featuremaps_1, sizeof(float) * featuremaps_1_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_1\n";
	}

	float* featuremaps_1_pooling;
	if (cudaMalloc((void**)&featuremaps_1_pooling, sizeof(float) * featuremaps_1_pooling_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_1_pooling\n";
	}

	float* featuremaps_2;
	if (cudaMalloc((void**)&featuremaps_2, sizeof(float) * featuremaps_2_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_2\n";
	}

	float* featuremaps_2_pooling;
	if (cudaMalloc((void**)&featuremaps_2_pooling, sizeof(float) * featuremaps_2_pooling_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_2_pooling\n";
	}

	float* featuremaps_3;
	if (cudaMalloc((void**)&featuremaps_3, sizeof(float) * featuremaps_3_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_3\n";
	}

	float* featuremaps_4;
	if (cudaMalloc((void**)&featuremaps_4, sizeof(float) * featuremaps_4_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_4\n";
	}


	dim3 grid_featuremaps_1(featuremaps_1_greed_size_x, featuremaps_1_greed_size_y, featuremaps_1_greed_size_z);
	dim3 threadBlock_featuremaps_1(featuremaps_1_thread_size_x, featuremaps_1_thread_size_y, featuremaps_1_thread_size_z);

	dim3 grid_featuremaps_1_pooling(featuremaps_1_pooling_greed_size_x, featuremaps_1_pooling_greed_size_y, featuremaps_1_pooling_greed_size_z);
	dim3 threadBlock_featuremaps_1_pooling(featuremaps_1_pooling_thread_size_x, featuremaps_1_pooling_thread_size_y, featuremaps_1_pooling_thread_size_z);

	dim3 grid_featuremaps_2(featuremaps_2_greed_size_x, featuremaps_2_greed_size_y, featuremaps_2_greed_size_z);
	dim3 threadBlock_featuremaps_2(featuremaps_2_thread_size_x, featuremaps_2_thread_size_y, featuremaps_2_thread_size_z);

	dim3 grid_featuremaps_2_pooling(featuremaps_2_pooling_greed_size_x, featuremaps_2_pooling_greed_size_y, featuremaps_2_pooling_greed_size_z);
	dim3 threadBlock_featuremaps_2_pooling(featuremaps_2_pooling_thread_size_x, featuremaps_2_pooling_thread_size_y, featuremaps_2_pooling_thread_size_z);

	dim3 grid_featuremaps_3(featuremaps_3_greed_size_x, featuremaps_3_greed_size_y, featuremaps_3_greed_size_z);
	dim3 threadBlock_featuremaps_3(featuremaps_3_thread_size_x, featuremaps_3_thread_size_y, featuremaps_3_thread_size_z);

	dim3 grid_featuremaps_4(featuremaps_4_greed_size_x, featuremaps_4_greed_size_y, featuremaps_4_greed_size_z);
	dim3 threadBlock_featuremaps_4(featuremaps_4_thread_size_x, featuremaps_4_thread_size_y, featuremaps_4_thread_size_z);


	//dim3 grid_featuremaps_1(2, 2, 4);
	//dim3 threadBlock_featuremaps_1(13, 13, 4);

	//printf("featuremaps_1_size = %d\n", featuremaps_1_size);
	//printf("sizeof(featuremaps_1) = %d\n", sizeof(featuremaps_1));
	//printf("sizeof(W_conv1) = %d\n", sizeof(W_conv1));
	//printf("sizeof(b_conv1) = %d\n", sizeof(b_conv1));
	//size_t b_conv1_sized = 0;
	//cudaError_t er1 = cudaGetSymbolSize(&b_conv1_sized, b_conv1);
	//printf("b_conv1_sized = %d\n", b_conv1_sized);
	//size_t featuremaps_1_sized = 0;
	//cudaError_t er2 = cudaGetSymbolSize(&featuremaps_1_sized, featuremaps_1);
	//printf("featuremaps_1_sized = %d\n", featuremaps_1_sized);

	//dim3 grid_featuremaps_1(1, 1, 1);
	//dim3 threadBlock_featuremaps_1(1, 1, 1);


	
	

	float* featuremaps_4_tmp_cpu;
	featuremaps_4_tmp_cpu = (float*)malloc(sizeof(float) * featuremaps_4_size);
	float featuremaps_4_max = 0.0;
	int featuremaps_4_max_ind = -1;
	int n_correct_answers = 0;
	clock_t begin = clock();
	for (int sample_count = 0; sample_count < n_samples; sample_count++)
	{
		
		


		int input_offset = sample_count * input_step;


		convolutions_relu_constants_weights_shared_memory<<<grid_featuremaps_1, threadBlock_featuremaps_1>>>(input_offset, x_val, input_size_x, input_step, input_n_channels, input_size,
			featuremaps_1, featuremaps_1_size_x, featuremaps_1_size_y_x, featuremaps_1_n_channels,
			offset_W_conv1, W_conv1_size_x, W_conv1_size_y, W_conv1_step_1, W_conv1_step_2, 
			offset_b_conv1,
			featuremaps_1_thread_size, featuremaps_1_threds_step_y_x);

		
		cudaDeviceSynchronize();

		
		max_pooling_2x2<<<grid_featuremaps_1_pooling, threadBlock_featuremaps_1_pooling>>> (featuremaps_1, featuremaps_1_size_x, featuremaps_1_size_y_x, featuremaps_1_n_channels,
			featuremaps_1_pooling, featuremaps_1_pooling_size_x, featuremaps_1_pooling_size_y_x);

		cudaDeviceSynchronize();

		convolutions_relu_constants_weights_shared_memory<<<grid_featuremaps_2, threadBlock_featuremaps_2>>>(0, featuremaps_1_pooling, featuremaps_1_pooling_size_x, featuremaps_1_pooling_size_y_x, featuremaps_1_pooling_n_channels, featuremaps_1_pooling_size,
			featuremaps_2, featuremaps_2_size_x, featuremaps_2_size_y_x, featuremaps_2_n_channels,
			offset_W_conv2, W_conv2_size_x, W_conv2_size_y, W_conv2_step_1, W_conv2_step_2,
			offset_b_conv2, 
			featuremaps_2_thread_size, featuremaps_2_threds_step_y_x);

		cudaDeviceSynchronize();

		max_pooling_3x3<<<grid_featuremaps_2_pooling, threadBlock_featuremaps_2_pooling>>>(featuremaps_2, featuremaps_2_size_x, featuremaps_2_size_y_x, featuremaps_2_n_channels,
			featuremaps_2_pooling, featuremaps_2_pooling_size_x, featuremaps_2_pooling_size_y_x);

		cudaDeviceSynchronize();


		convolutions_relu_shared_memory<<<grid_featuremaps_3, threadBlock_featuremaps_3>>> (0, featuremaps_2_pooling, featuremaps_2_pooling_size_x, featuremaps_2_pooling_size_y_x, featuremaps_2_pooling_n_channels, featuremaps_2_pooling_size,
			featuremaps_3, featuremaps_3_size_x, featuremaps_3_size_y_x, featuremaps_3_n_channels,
			W_conv3, W_conv3_size_x, W_conv3_size_y, W_conv3_step_1, W_conv3_step_2,
			b_conv3,
			featuremaps_3_thread_size, featuremaps_3_threds_step_y_x);

		cudaDeviceSynchronize();


		convolutions_shared_memory<<<grid_featuremaps_4, threadBlock_featuremaps_4>>>(0, featuremaps_3, featuremaps_3_size_x, featuremaps_3_size_y_x, featuremaps_3_n_channels, featuremaps_3_size, 
			featuremaps_4, featuremaps_4_size_x, featuremaps_4_size_y_x, featuremaps_4_n_channels,
			W_conv4, W_conv4_size_x, W_conv4_size_y, W_conv4_step_1, W_conv4_step_2,
			b_conv4,
			featuremaps_4_thread_size, featuremaps_4_threds_step_y_x);

		cudaDeviceSynchronize();

		cudaMemcpy(featuremaps_4_tmp_cpu, featuremaps_4, sizeof(float)* featuremaps_4_size, cudaMemcpyDeviceToHost);

		
		featuremaps_4_max = featuremaps_4_tmp_cpu[0];
		featuremaps_4_max_ind = 0;
		for (int output_index = 1; output_index < n_output; output_index++)
		{
			//printf("output_index = %d\n", output_index);
			if (featuremaps_4_tmp_cpu[output_index] > featuremaps_4_max)
			{
				featuremaps_4_max = featuremaps_4_tmp_cpu[output_index];
				featuremaps_4_max_ind = output_index;
				//printf("featuremaps_4_max = %.6fd\n", featuremaps_4_max);
				//printf("featuremaps_4_max_ind = %d\n", featuremaps_4_max_ind);
			}
		}
		//printf("featuremaps_4_max_ind =%d\n", featuremaps_4_max_ind);
		//printf("y_val_cpu[sample_count] =%d\n", y_val_cpu[sample_count]);
		if (featuremaps_4_max_ind == y_val_cpu[sample_count])
		{
			n_correct_answers++;
		}

		

		
			
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	double time_mean = elapsed_secs / n_samples;

	float accuracy = ((float)n_correct_answers) / n_samples;
	printf("accuracy = %.8f\n", accuracy);
	printf("elapsed_secs = %.8f\n", elapsed_secs);
	printf("time_mean = %.8f\n", time_mean);
	
	

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "last ERROR after check almost at end: %s\n", cudaGetErrorString(error));
	}

	
	
	
	
	
	
	free(featuremaps_4_tmp_cpu);
	free(y_val_cpu);

	
	
	cudaFree(x_val);
	cudaFree(featuremaps_1);
	cudaFree(featuremaps_1_pooling);
	cudaFree(featuremaps_2);
	cudaFree(featuremaps_2_pooling);
	cudaFree(featuremaps_3);
	cudaFree(featuremaps_4);

	
	cudaFree(W_conv3);
	cudaFree(b_conv3);
	cudaFree(W_conv4);
	cudaFree(b_conv4);
	
	
	
	
	return 0;


}