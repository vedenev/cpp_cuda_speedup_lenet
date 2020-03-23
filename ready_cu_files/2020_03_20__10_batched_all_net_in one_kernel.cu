

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



__global__ void net_batch(float* x_val, int input_size_x, int input_step, int input_n_channels, int input_size_1_sample,
	float* featuremaps_1, int featuremaps_1_size_x, int featuremaps_1_size_y, int featuremaps_1_size_y_x, int featuremaps_1_n_channels, int featuremaps_1_size_1_sample, 
	float* featuremaps_1_pooling, int featuremaps_1_pooling_size_x, int featuremaps_1_pooling_size_y, int featuremaps_1_pooling_size_y_x, int featuremaps_1_pooling_n_channels, int featuremaps_1_pooling_size_1_sample,
	int offset_W_conv1, int W_conv1_size_x, int W_conv1_size_y, int W_conv1_step_1, int W_conv1_step_2,
	int offset_b_conv1,

	float* featuremaps_2, int featuremaps_2_size_x, int featuremaps_2_size_y, int featuremaps_2_size_y_x, int featuremaps_2_n_channels, int featuremaps_2_size_1_sample,
	float* featuremaps_2_pooling, int featuremaps_2_pooling_size_x, int featuremaps_2_pooling_size_y, int featuremaps_2_pooling_size_y_x, int featuremaps_2_pooling_n_channels, int featuremaps_2_pooling_size, int featuremaps_2_pooling_size_1_sample,
	int offset_W_conv2, int W_conv2_size_x, int W_conv2_size_y, int W_conv2_step_1, int W_conv2_step_2,
	int offset_b_conv2,

	float* featuremaps_3, int featuremaps_3_size_x, int featuremaps_3_size_y, int featuremaps_3_size_y_x, int featuremaps_3_n_channels, int featuremaps_3_size_1_sample,
	float* W_conv3, int W_conv3_size_x, int W_conv3_size_y, int W_conv3_step_1, int W_conv3_step_2,
	float* b_conv3,


	int featuremaps_4_size_x, int featuremaps_4_size_y, int featuremaps_4_size_y_x, int featuremaps_4_n_channels, int featuremaps_4_size_1_sample,
	float* W_conv4, int W_conv4_size_x, int W_conv4_size_y, int W_conv4_step_1, int W_conv4_step_2,
	float* b_conv4,
	
	char* y_val)

{ 

	int output_1d_offset;
	const unsigned int in_batch_index = blockIdx.x * blockDim.x + threadIdx.x;
	int input_offset;

	// conv1:
	input_offset = input_size_1_sample * in_batch_index;
	output_1d_offset = featuremaps_1_size_1_sample * in_batch_index;
	for (int index_output_channel = 0; index_output_channel < featuremaps_1_n_channels; index_output_channel++)
	{
		unsigned int weights_1d_index_offset_0 = W_conv1_step_2 * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_1_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_1_size_y; index_output_y++)
		{
			int output_1d_index_2 = output_1d_index_1 + featuremaps_1_size_x * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_1_size_x; index_output_x++)
			{
				int output_1d_index = output_1d_index_2 + index_output_x;
				float output_value = 0.0;
				for (int index_input_channel = 0; index_input_channel < input_n_channels; index_input_channel++)
				{
					//W_conv1_step_1 = W_conv1_size_x * W_conv1_size_y;
					//W_conv1_step_2 = W_conv1_size_x * W_conv1_size_y * input_n_channels;
					// W index: (W_conv1_size_x * W_conv1_size_y * input_n_channels) * index_output_channel + (W_conv1_size_x * W_conv1_size_y) * index_input_channel + (W_conv1_size_x) * weights_index_y + weights_index_x
					unsigned int weights_1d_index_offset = weights_1d_index_offset_0 + W_conv1_step_1 * index_input_channel;
					for (int weights_index_y = 0; weights_index_y < W_conv1_size_y; weights_index_y++) {
						for (int weights_index_x = 0; weights_index_x < W_conv1_size_x; weights_index_x++) {
							unsigned int index_input_x = index_output_x + weights_index_x;
							unsigned int index_input_y = index_output_y + weights_index_y;
							unsigned int input_1d_index = input_step * index_input_channel +
								input_size_x * index_input_y + index_input_x;
							//input_step = input_size_x * input_size_y; // 784
							unsigned int weights_1d_index = weights_1d_index_offset + W_conv1_size_x * weights_index_y + weights_index_x;
							output_value += x_val[input_offset + input_1d_index] * constants[offset_W_conv1 + weights_1d_index];
							//output_value += x_val[input_offset + input_1d_index];
							//output_value += constants[offset_W_conv1 + weights_1d_index];
							//output_value += input_offset + input_1d_index;
							//output_value = input_offset + input_1d_index;
							//output_value += 1;
							//output_value = in_batch_index;
							//output_value = input_offset;
							//output_value = output_1d_offset;
							//output_value = input_1d_index;
							//output_value = input_size;
							//output_value = index_output_y;
							//output_value = featuremaps_1_size_y_x;


						}
					}

				}
				output_value += constants[offset_b_conv1 + index_output_channel];
				output_value = fmaxf(output_value, 0.0); // relu
				featuremaps_1[output_1d_index] = output_value;
				//featuremaps_1[output_1d_index] = 6.28;
				//featuremaps_1[output_1d_index] = featuremaps_1_size_1_sample;
			}
		}
	}
	
	// pool1:
	input_offset = output_1d_offset;
	output_1d_offset = featuremaps_1_pooling_size_1_sample * in_batch_index;
	for (int index_output_channel = 0; index_output_channel < featuremaps_1_pooling_n_channels; index_output_channel++)
	{
		unsigned int features_input_step = input_offset + featuremaps_1_size_y_x * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_1_pooling_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_1_pooling_size_y; index_output_y++)
		{
			int output_1d_index_2 = output_1d_index_1 + featuremaps_1_pooling_size_x * index_output_y;
			unsigned int index_input_y = 2 * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_1_pooling_size_x; index_output_x++)
			{
				unsigned int index_input_x = 2 * index_output_x;
				
				unsigned int output_1d_index = output_1d_index_2 + index_output_x;

				
				unsigned int input_1d_index_0_0 = features_input_step +
					featuremaps_1_size_x * index_input_y + index_input_x;
				unsigned int input_1d_index_0_1 = input_1d_index_0_0 + 1;
				unsigned int input_1d_index_1_0 = input_1d_index_0_0 + featuremaps_1_size_x;
				unsigned int input_1d_index_1_1 = input_1d_index_0_0 + 1 + featuremaps_1_size_x;

				float max_0 = fmaxf(featuremaps_1[input_1d_index_0_0], featuremaps_1[input_1d_index_0_1]);
				float max_1 = fmaxf(featuremaps_1[input_1d_index_1_0], featuremaps_1[input_1d_index_1_1]);
				featuremaps_1_pooling[output_1d_index] = fmaxf(max_0, max_1);
			}
		}
	}


	// conv2:
	input_offset = output_1d_offset;
	output_1d_offset = featuremaps_2_size_1_sample * in_batch_index;
	for (int index_output_channel = 0; index_output_channel < featuremaps_2_n_channels; index_output_channel++)
	{
		unsigned int weights_1d_index_offset_0 = W_conv2_step_2 * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_2_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_2_size_y; index_output_y++)
		{
			int output_1d_index_2 = output_1d_index_1 + featuremaps_2_size_x * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_2_size_x; index_output_x++)
			{
				int output_1d_index = output_1d_index_2 + index_output_x;
				float output_value = 0.0;
				for (int index_input_channel = 0; index_input_channel < featuremaps_1_pooling_n_channels; index_input_channel++)
				{
					unsigned int weights_1d_index_offset = weights_1d_index_offset_0 + W_conv2_step_1 * index_input_channel;
					for (int weights_index_y = 0; weights_index_y < W_conv2_size_y; weights_index_y++) {
						for (int weights_index_x = 0; weights_index_x < W_conv2_size_x; weights_index_x++) {
							unsigned int index_input_x = index_output_x + weights_index_x;
							unsigned int index_input_y = index_output_y + weights_index_y;
							unsigned int input_1d_index = featuremaps_1_pooling_size_y_x * index_input_channel +
								featuremaps_1_pooling_size_x * index_input_y + index_input_x;
							unsigned int weights_1d_index = weights_1d_index_offset + W_conv2_size_x * weights_index_y + weights_index_x;
							output_value += featuremaps_1_pooling[input_offset + input_1d_index] * constants[offset_W_conv2 + weights_1d_index];
						}
					}

				}
				output_value += constants[offset_b_conv2 + index_output_channel];
				output_value = fmaxf(output_value, 0.0); // relu
				featuremaps_2[output_1d_index] = output_value;
			}
		}
	}

	// pool2:
	input_offset = output_1d_offset;
	output_1d_offset = featuremaps_2_pooling_size_1_sample * in_batch_index;
	for (int index_output_channel = 0; index_output_channel < featuremaps_2_pooling_n_channels; index_output_channel++)
	{
		unsigned int features_input_step = input_offset + featuremaps_2_size_y_x * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_2_pooling_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_2_pooling_size_y; index_output_y++)
		{
			unsigned int index_input_y = 3 * index_output_y;
			int output_1d_index_2 = output_1d_index_1 + featuremaps_2_pooling_size_x * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_2_pooling_size_x; index_output_x++)
			{
				unsigned int index_input_x = 3 * index_output_x;
				

				unsigned int output_1d_index = output_1d_index_2 + index_output_x;

				unsigned int input_1d_index_0_0 = features_input_step +
					featuremaps_2_size_x * index_input_y + index_input_x;
				unsigned int input_1d_index_0_1 = input_1d_index_0_0 + 1;
				unsigned int input_1d_index_0_2 = input_1d_index_0_0 + 2;
				unsigned int input_1d_index_1_0 = input_1d_index_0_0 + featuremaps_2_size_x;
				unsigned int input_1d_index_1_1 = input_1d_index_1_0 + 1;
				unsigned int input_1d_index_1_2 = input_1d_index_1_0 + 2;
				unsigned int input_1d_index_2_0 = input_1d_index_1_0 + featuremaps_2_size_x;
				unsigned int input_1d_index_2_1 = input_1d_index_2_0 + 1;
				unsigned int input_1d_index_2_2 = input_1d_index_2_0 + 2;

				float max_0 = fmaxf(featuremaps_2[input_1d_index_0_0], featuremaps_2[input_1d_index_0_1]);
				float max_1 = fmaxf(featuremaps_2[input_1d_index_0_2], featuremaps_2[input_1d_index_1_0]);
				float max_2 = fmaxf(featuremaps_2[input_1d_index_1_1], featuremaps_2[input_1d_index_1_2]);
				float max_3 = fmaxf(featuremaps_2[input_1d_index_2_0], featuremaps_2[input_1d_index_2_1]);

				float max_4 = fmaxf(max_0, max_1);
				float max_5 = fmaxf(max_2, max_3);

				float max_6 = fmaxf(max_4, max_5);

				featuremaps_2_pooling[output_1d_index] = fmaxf(max_6, featuremaps_2[input_1d_index_2_2]);
			}
		}
	}


	// conv3:
	input_offset = output_1d_offset;
	output_1d_offset = featuremaps_3_size_1_sample * in_batch_index;
	for (int index_output_channel = 0; index_output_channel < featuremaps_3_n_channels; index_output_channel++)
	{
		unsigned int weights_1d_index_offset_0 = W_conv3_step_2 * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_3_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_3_size_y; index_output_y++)
		{
			int output_1d_index_2 = output_1d_index_1 + featuremaps_3_size_x * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_3_size_x; index_output_x++)
			{
				int output_1d_index = output_1d_index_2 + index_output_x;
				float output_value = 0.0;
				for (int index_input_channel = 0; index_input_channel < featuremaps_2_pooling_n_channels; index_input_channel++)
				{
					unsigned int weights_1d_index_offset = weights_1d_index_offset_0 + W_conv3_step_1 * index_input_channel;
					for (int weights_index_y = 0; weights_index_y < W_conv3_size_y; weights_index_y++) {
						for (int weights_index_x = 0; weights_index_x < W_conv3_size_x; weights_index_x++) {
							unsigned int index_input_x = index_output_x + weights_index_x;
							unsigned int index_input_y = index_output_y + weights_index_y;
							unsigned int input_1d_index = featuremaps_2_pooling_size_y_x * index_input_channel +
								featuremaps_2_pooling_size_x * index_input_y + index_input_x;
							unsigned int weights_1d_index = weights_1d_index_offset + W_conv3_size_x * weights_index_y + weights_index_x;
							output_value += featuremaps_2_pooling[input_offset + input_1d_index] * W_conv3[weights_1d_index];
						}
					}

				}
				output_value += b_conv3[index_output_channel];
				output_value = fmaxf(output_value, 0.0); // relu
				featuremaps_3[output_1d_index] = output_value;
			}
		}
	}


	// conv4 + max:
	input_offset = output_1d_offset;
	output_1d_offset =  featuremaps_4_size_1_sample * in_batch_index;
	int max_index = -1;
	float max_value = -1e30;
	for (int index_output_channel = 0; index_output_channel < featuremaps_4_n_channels; index_output_channel++)
	{
		unsigned int weights_1d_index_offset_0 = W_conv4_step_2 * index_output_channel;
		int output_1d_index_1 = output_1d_offset + featuremaps_4_size_y_x * index_output_channel;
		for (int index_output_y = 0; index_output_y < featuremaps_4_size_y; index_output_y++)
		{
			int output_1d_index_2 = output_1d_index_1 + featuremaps_4_size_x * index_output_y;
			for (int index_output_x = 0; index_output_x < featuremaps_4_size_x; index_output_x++)
			{
				int output_1d_index = output_1d_index_2 + index_output_x;
				float output_value = 0.0;
				for (int index_input_channel = 0; index_input_channel < featuremaps_3_n_channels; index_input_channel++)
				{
					unsigned int weights_1d_index_offset = weights_1d_index_offset_0 + W_conv4_step_1 * index_input_channel;
					for (int weights_index_y = 0; weights_index_y < W_conv4_size_y; weights_index_y++) {
						for (int weights_index_x = 0; weights_index_x < W_conv4_size_x; weights_index_x++) {
							unsigned int index_input_x = index_output_x + weights_index_x;
							unsigned int index_input_y = index_output_y + weights_index_y;
							unsigned int input_1d_index = featuremaps_3_size_y_x * index_input_channel +
								featuremaps_3_size_x * index_input_y + index_input_x;
							unsigned int weights_1d_index = weights_1d_index_offset + W_conv4_size_x * weights_index_y + weights_index_x;
							output_value += featuremaps_3[input_offset + input_1d_index] * W_conv4[weights_1d_index];
						}
					}

				}
				output_value += b_conv4[index_output_channel];
				if (output_value > max_value)
				{
					max_value = output_value;
					max_index = index_output_channel;
				}
			}
		}
	}


	y_val[in_batch_index] = max_index;

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
	
	
	int batch_size = 3072;

	int input_size_x = 28;
	int input_size_y = 28;
	int input_n_channels = 1;
	int n_output = 10;

	int input_step;
	int input_size;

	int featuremaps_1_size_x = 26;
	int featuremaps_1_size_y = 26;
	int featuremaps_1_size_y_x;
	int featuremaps_1_n_channels = 16;
	int featuremaps_1_size;


	int featuremaps_1_pooling_size_x = 13;
	int featuremaps_1_pooling_size_y = 13;
	int featuremaps_1_pooling_n_channels;
	int featuremaps_1_pooling_size_y_x;
	int featuremaps_1_pooling_size;


	int featuremaps_2_size_x = 9;
	int featuremaps_2_size_y = 9;
	int featuremaps_2_size_y_x;
	int featuremaps_2_n_channels = 16;
	int featuremaps_2_size;


	int featuremaps_2_pooling_size_x = 3;
	int featuremaps_2_pooling_size_y = 3;
	int featuremaps_2_pooling_n_channels;
	int featuremaps_2_pooling_size_y_x;
	int featuremaps_2_pooling_size;

	int featuremaps_3_size_x = 1;
	int featuremaps_3_size_y = 1;
	int featuremaps_3_n_channels = 256;
	int featuremaps_3_size;

	int featuremaps_4_size_x = 1;
	int featuremaps_4_size_y = 1;
	int featuremaps_4_n_channels;
	int featuremaps_4_size;
	int featuremaps_4_size_y_x;


	int W_conv1_size_x = 3;
	int W_conv1_size_y = 3;
	int W_conv1_size;
	int W_conv1_step_1;
	int W_conv1_step_2;
	int b_conv1_size;

	int W_conv2_size_x = 5;
	int W_conv2_size_y = 5;
	int W_conv2_size;
	int W_conv2_step_1;
	int W_conv2_step_2;
	int b_conv2_size;

	int W_conv3_size_x = 3;
	int W_conv3_size_y = 3;
	int W_conv3_size;
	int W_conv3_step_1;
	int W_conv3_step_2;
	int b_conv3_size;

	int W_conv4_size_x = 1;
	int W_conv4_size_y = 1;
	int W_conv4_size;
	int W_conv4_step_1;
	int W_conv4_step_2;
	int b_conv4_size = 10;

	int x_val_size = 7840000;
	int n_samples = 10000;




	float* W_conv3;
	float* b_conv3;
	float* W_conv4;
	float* b_conv4;
	float* x_val;

	float* featuremaps_1;
	float* featuremaps_1_pooling;
	float* featuremaps_2;
	float* featuremaps_2_pooling;
	float* featuremaps_3;
	//float* featuremaps_4;

	int input_size_1_sample;
	int featuremaps_1_size_1_sample;
	int featuremaps_1_pooling_size_1_sample;
	int featuremaps_2_size_1_sample;
	int featuremaps_2_pooling_size_1_sample;
	int featuremaps_3_size_y_x;
	int featuremaps_3_size_1_sample;
	int featuremaps_4_size_1_sample;

	

	//char* predictions;
	char* y_val;
	

	char* weights_dir = "F:/freelance/cpp_learning/cuda_learning/weigths_1d/";
	



	

	//              c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
	// 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
	// n mult : 97344                                 518400                          36864             2560
	// 784                   10816           2704               1296              144                256            10
	
	

	input_step = input_size_x * input_size_y;
	input_size = input_size_x * input_size_y * input_n_channels * batch_size;


	featuremaps_1_size_y_x = featuremaps_1_size_x * featuremaps_1_size_y;
	featuremaps_1_size = featuremaps_1_size_x * featuremaps_1_size_y * featuremaps_1_n_channels * batch_size;
	

	featuremaps_1_pooling_n_channels = featuremaps_1_n_channels;
	featuremaps_1_pooling_size_y_x = featuremaps_1_pooling_size_x * featuremaps_1_pooling_size_y;
	featuremaps_1_pooling_size = featuremaps_1_pooling_size_x * featuremaps_1_pooling_size_y * featuremaps_1_n_channels * batch_size;
	
	featuremaps_2_size_y_x = featuremaps_2_size_x * featuremaps_2_size_y;
	featuremaps_2_size = featuremaps_2_size_x * featuremaps_2_size_y * featuremaps_2_n_channels * batch_size;
	
	featuremaps_2_pooling_n_channels = featuremaps_2_n_channels;
	featuremaps_2_pooling_size_y_x = featuremaps_2_pooling_size_x * featuremaps_2_pooling_size_y;
	featuremaps_2_pooling_size = featuremaps_2_pooling_size_x * featuremaps_2_pooling_size_y * featuremaps_2_n_channels * batch_size;
	
	featuremaps_3_size = featuremaps_3_size_x * featuremaps_3_size_y * featuremaps_3_n_channels * batch_size;
	
	
	featuremaps_4_n_channels = n_output;
	featuremaps_4_size = featuremaps_4_size_x * featuremaps_4_size_y * featuremaps_4_n_channels * batch_size;
	featuremaps_4_size_y_x = featuremaps_4_size_x * featuremaps_4_size_y;
	

	input_size_1_sample = input_size_x * input_size_y * input_n_channels;
	featuremaps_1_size_1_sample = featuremaps_1_size_x * featuremaps_1_size_y * featuremaps_1_n_channels;
	featuremaps_1_pooling_size_1_sample = featuremaps_1_pooling_size_x * featuremaps_1_pooling_size_y * featuremaps_1_n_channels;
	featuremaps_2_size_1_sample = featuremaps_2_size_x * featuremaps_2_size_y * featuremaps_2_n_channels;
	featuremaps_2_pooling_size_1_sample = featuremaps_2_pooling_size_x * featuremaps_2_pooling_size_y * featuremaps_2_n_channels;
	featuremaps_3_size_y_x = featuremaps_3_size_x * featuremaps_3_size_y;
	featuremaps_3_size_1_sample = featuremaps_3_size_x * featuremaps_3_size_y * featuremaps_3_n_channels;
	featuremaps_4_size_1_sample = featuremaps_4_size_x * featuremaps_4_size_y * featuremaps_4_n_channels;


	W_conv1_size = W_conv1_size_x * W_conv1_size_y * input_n_channels * featuremaps_1_n_channels;
	W_conv1_step_1 = W_conv1_size_x * W_conv1_size_y;
	W_conv1_step_2 = W_conv1_size_x * W_conv1_size_y * input_n_channels;
	b_conv1_size = featuremaps_1_n_channels;

	W_conv2_size = W_conv2_size_x * W_conv2_size_y * featuremaps_1_n_channels * featuremaps_2_n_channels;
	W_conv2_step_1 = W_conv2_size_x * W_conv2_size_y;
	W_conv2_step_2 = W_conv2_size_x * W_conv2_size_y * featuremaps_1_n_channels;
	b_conv2_size = featuremaps_2_n_channels;

	W_conv3_size = W_conv3_size_x * W_conv3_size_y * featuremaps_2_n_channels * featuremaps_3_n_channels;
	W_conv3_step_1 = W_conv3_size_x * W_conv3_size_y;
	W_conv3_step_2 = W_conv3_size_x * W_conv3_size_y * featuremaps_2_n_channels;
	b_conv3_size = featuremaps_3_n_channels;

	W_conv4_size = W_conv4_size_x * W_conv4_size_y * featuremaps_3_n_channels * featuremaps_4_n_channels;
	W_conv4_step_1 = W_conv4_size_x * W_conv4_size_y;
	W_conv4_step_2 = W_conv4_size_x * W_conv4_size_y * featuremaps_3_n_channels;
	

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
	//load_data_to_array(weights_dir, "x_val.bin", &x_val, x_val_size);
	
	char* y_val_cpu;
	y_val_cpu = (char*)malloc(sizeof(char) * n_samples);
	FILE* file_id;
	char* path;
	c_stings_concatinate(weights_dir, "y_val.bin", &path);
	file_id = fopen(path, "rb");
	fread(y_val_cpu, sizeof(char), n_samples, file_id);
	fclose(file_id);
	free(path);

	char* y_val_cpu_prediction;
	y_val_cpu_prediction = (char*)malloc(sizeof(char) * batch_size);


	float* x_val_cpu;
	x_val_cpu = (float*)malloc(sizeof(float) * x_val_size);
	c_stings_concatinate(weights_dir, "x_val.bin", &path);
	file_id = fopen(path, "rb");
	fread(x_val_cpu, sizeof(float), x_val_size, file_id);
	fclose(file_id);
	free(path);

	// check x_val_cpu by calculating mean value:
	float x_val_cpu_sum = 0.0;
	for (int pixel_index = 0; pixel_index < x_val_size; pixel_index++)
	{
		x_val_cpu_sum += x_val_cpu[pixel_index];
	}
	float x_val_cpu_mean = x_val_cpu_sum / x_val_size;
	printf("x_val_cpu_mean = %.8f\n", x_val_cpu_mean);
	
	if (cudaMalloc((void**)&featuremaps_1, sizeof(float) * featuremaps_1_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_1\n";
	}

	
	if (cudaMalloc((void**)&featuremaps_1_pooling, sizeof(float) * featuremaps_1_pooling_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_1_pooling\n";
	}

	
	if (cudaMalloc((void**)&featuremaps_2, sizeof(float) * featuremaps_2_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_2\n";
	}

	
	if (cudaMalloc((void**)&featuremaps_2_pooling, sizeof(float) * featuremaps_2_pooling_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_2_pooling\n";
	}

	
	if (cudaMalloc((void**)&featuremaps_3, sizeof(float) * featuremaps_3_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU featuremaps_3\n";
	}

	
	//if (cudaMalloc((void**)&featuremaps_4, sizeof(float) * featuremaps_4_size) != cudaSuccess)
	//{
	//	std::cout << "Error allocating GPU featuremaps_4\n";
	//}


	//if (cudaMalloc((void**)&predictions, sizeof(char) * batch_size) != cudaSuccess)
	//{
	//	std::cout << "Error allocating GPU predictions\n";
	//}

	if (cudaMalloc((void**)&y_val, sizeof(float) * batch_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU y_val\n";
	}
	
	if (cudaMalloc((void**)&x_val, sizeof(float) * input_size) != cudaSuccess)
	{
		std::cout << "Error allocating GPU x_val\n";
	}
	
	int n_batches = n_samples / batch_size;
	
	int n_threads;
	int n_blocks;

	if (batch_size <= 1024)
	{
		n_threads = batch_size;
		n_blocks = batch_size / n_threads; // = 1
	}
	else
	{
		n_threads = 1024;
		n_blocks = batch_size / n_threads;
	}
	
	
	
	int n_correct_answers = 0;
	int sample_count_global = 0;
	int n_answers = 0;
	clock_t begin = clock();
	for (int batch_count = 0; batch_count < n_batches; batch_count++)
	{
		
		


		int input_offset = batch_count * input_size;
		cudaMemcpy(x_val, (x_val_cpu + input_offset), sizeof(float) * input_size, cudaMemcpyHostToDevice);
		//cudaDeviceSynchronize();
		net_batch<<<n_blocks, n_threads>>>(x_val, input_size_x, input_step, input_n_channels, input_size_1_sample,
			featuremaps_1, featuremaps_1_size_x, featuremaps_1_size_y, featuremaps_1_size_y_x, featuremaps_1_n_channels, featuremaps_1_size_1_sample,
			featuremaps_1_pooling, featuremaps_1_pooling_size_x, featuremaps_1_pooling_size_y, featuremaps_1_pooling_size_y_x, featuremaps_1_pooling_n_channels, featuremaps_1_pooling_size_1_sample,
			offset_W_conv1, W_conv1_size_x, W_conv1_size_y, W_conv1_step_1, W_conv1_step_2,
			offset_b_conv1,

			featuremaps_2, featuremaps_2_size_x, featuremaps_2_size_y, featuremaps_2_size_y_x, featuremaps_2_n_channels, featuremaps_2_size_1_sample,
			featuremaps_2_pooling, featuremaps_2_pooling_size_x, featuremaps_2_pooling_size_y, featuremaps_2_pooling_size_y_x, featuremaps_2_pooling_n_channels, featuremaps_2_pooling_size, featuremaps_2_pooling_size_1_sample,
			offset_W_conv2, W_conv2_size_x, W_conv2_size_y, W_conv2_step_1, W_conv2_step_2,
			offset_b_conv2,

			featuremaps_3, featuremaps_3_size_x, featuremaps_3_size_y, featuremaps_3_size_y_x, featuremaps_3_n_channels, featuremaps_3_size_1_sample,
			W_conv3, W_conv3_size_x, W_conv3_size_y, W_conv3_step_1, W_conv3_step_2,
			b_conv3,

			
			featuremaps_4_size_x, featuremaps_4_size_y, featuremaps_4_size_y_x, featuremaps_4_n_channels, featuremaps_4_size_1_sample,
			W_conv4, W_conv4_size_x, W_conv4_size_y, W_conv4_step_1, W_conv4_step_2,
			b_conv4,

			y_val);

		cudaDeviceSynchronize();
		cudaMemcpy(y_val_cpu_prediction, y_val, sizeof(char) * batch_size, cudaMemcpyDeviceToHost);
		int output_offset = batch_count * batch_size;
		for (int sample_count = 0; sample_count < batch_size; sample_count++)
		{
			
			if (y_val_cpu_prediction[sample_count] == y_val_cpu[output_offset + sample_count])
			{
				n_correct_answers++;
			}
			n_answers++;
			sample_count_global++;
		}

		

		
			
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	double time_mean = elapsed_secs / n_answers;

	float accuracy = ((float)n_correct_answers) / n_answers;
	printf("accuracy = %.8f\n", accuracy);
	printf("elapsed_secs = %.8f\n", elapsed_secs);
	printf("time_mean = %.8f\n", time_mean);
	printf("n_answers =%d", n_answers);
	
	

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "last ERROR after check almost at end: %s\n", cudaGetErrorString(error));
	}

	
	
	
	
	
	
	//free(featuremaps_4_tmp_cpu);
	free(y_val_cpu);
	free(y_val_cpu_prediction);

	
	
	cudaFree(x_val);
	cudaFree(featuremaps_1);
	cudaFree(featuremaps_1_pooling);
	cudaFree(featuremaps_2);
	cudaFree(featuremaps_2_pooling);
	cudaFree(featuremaps_3);
	//cudaFree(featuremaps_4);

	
	cudaFree(W_conv3);
	cudaFree(b_conv3);
	cudaFree(W_conv4);
	cudaFree(b_conv4);
	
	
	
	
	return 0;


}