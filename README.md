# Try to make cuda code that faster then tensorflow for lenet  

Here I made LeNet that recognizes MNIST. First I made it in Tensorflow then in CUDA (only inference). I tried to make optimizations in CUDA code to make it faster then Tensorflow inference. Here are final results:  
Tensorflow 1.4 vs CUDA on GPU GTX760:  
![tf vs cuda](/times_semilogy_tf_1_4_vs_cuda_gtx760__2020_03_23.png)  
Tensorflow 1.4 vs Tensorflow 1.15 on GPU Tesla K80:
![tf 1.4 vs tf 1.5](/times_semilogy_tf_1_4_vs_tf_1_15_k80__2020_03_23.png)  
Final conclusion: for this codes CUDA code is faster then Tensorflow 1.4 if batch size is 1, but for bigger but batch size tensorflow is faster.  
May be it is possible to accelerate CUDA codes more.  
cuda files, see ready_cu_files subfolder:  
 description | .cu file |  speed, sec/sample 
---|---|---  
all weights constants except W3_conv|2020_02_13__2_stable_maximal_constnats.cu|0.00127470
1st and 2nd layers have constant weights|2020_02_13__1_stable_constant_weights_1st_and_2nd_layers.cu|0.00118140
no constant memory|2020_02_13__3_stable_no_constant_weights.cu|0.00120700
1st and 2nd with constants, all rest with shared memeory, with repeated mutiply|2020_02_18__05_stable_shared_memory_all_layers.cu|0.00105580
1st and 2nd with constants, all rest with shared memeory, no repeated mutiply|2020_02_20__08_stable_no_repeated_multiply_memory_all_layers.cu|0.00104740
1st and 2nd with constants, all rest with shared memeory, no repeated mutiply, input from CPU|2020_02_27__09_stable_no_repeated_multiply_memory_all_layers_input_from_cpu.cu|0.00112120
batched, all net in one kernel, batch size = 512|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00136102
batched, all net in one kernel, batch size = 1024|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00126280
batched, all net in one kernel, batch size = 2048|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00064478
batched, all net in one kernel, batch size = 3072|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00047949
---|---|--- 
I used Microsoft Visual Studio 2019, free comunity edition and example vectorAdd (see vectorAdd subfolder) from Nvidia CUDA samples. So to use it copy one of this cu-files to the project directory and rename it to vectorAdd.cu. Also in code specify path to weigths_1d folder. See ```char* weights_dir =  ...``` in the code. How to run the sample:  
1) copy all  
from C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\visual_studio_integration\MSBuildExtensions  
to C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations  
https://devtalk.nvidia.com/default/topic/933708/cuda-setup-and-installation/compiling-and-setting-up-cuda-libraries-on-windows-10/post/5274566/#5274566  
2) Open 2015 project without conversion  
3) in properties in configuration in general change target platform version version from 10 to 8  
LeNet architecture used:  
```
architecture:            c 3 x 3            p 2 x 2           c 5 x 5         p 3 x 3         c 3 x 3          c 1 x 1
fieaturemaps: 28 x 28 x 1   ->   26 x 26 x 16   ->  13 x 13 x 16  ->  9 x 9 x 16 ->   3 x 3 x 16  ->  1 x 1 x 256  ->   1 x 1 x 10
n multiplies  :          97344                                 518400                          36864             2560
dimentions:      784                   10816           2704               1296              144                256            10
```
There is no pading here (no zero inputs somewhere in the middle of the net). Last 2 layers are actually fully connected. They are emulated with convolutional layers for easy convert to CUDA codes.  
First I trainied the net in Tensorflow 1.4 in python. Then convert dataset and weights in to binary format to be able to load them in C++ code, see weigths_1d folder.  
Python codes:  
| .py file | description |
|---|---|
|t002_write_test_arrays.py|Test writing arrays to binary files|
|t003_mnist.py|Test load MNIST dataset in Tensorflow|
|t005_train.py|Train LeNet in Tensorflow|
|t006_plot_losses.py|Plot losses saved in t005_train.py|
|t007_freeze_graph.py|Frize graph of trained net and save it to .pb file|
|t008_test_inference.py|Run inference on validation dataset in Tensorflow and get accuracy and speed|
|t009_freeze_graph_with_input_data_try1.py|Frize graph of trained net and save it to .pb file with validation dataset inside|
|t016_test_inference_batch_variable.py|Inference, test metrics vs batch size|
|t017_convert_weigths_to_binary_files.py|convert dataset and weights in to binary format to be able to load them in C++ code|