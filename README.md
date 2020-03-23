# try to make cuda code that faster then tensorflow for lenet  

Here I made LeNet that recognizes MNIST. First I made it in Tensorflow then in CUDA (only inference). I tried to make optimizations in CUDA code to make it faster then Tensorflow inference. Here are final results:  
Tensorflow 1.4 vs CUDA on GPU GTX760:  
![tf vs cuda](/times_semilogy_tf_1_4_vs_cuda_gtx760__2020_03_23.png)  
Tensorflow 1.4 vs Tensorflow 1.15 on GPU Tesla K80:
![tf 1.4 vs tf 1.5](/times_semilogy_tf_1_4_vs_tf_1_15_k80__2020_03_23.png)  
Final conclusion: for this codes CUDA code is faster then Tensorflow 1.4 if batch size is 1, but for bigger but batch size tensorflow is faster.  
| description | .cu file |  speed |
|---|---|---|  
|all weights constants except W3_conv|2020_02_13__2_stable_maximal_constnats.cu|0.00127470 sec/sample|
|1st and 2nd layers have constant weights|2020_02_13__1_stable_constant_weights_1st_and_2nd_layers.cu|0.00118140 sec/sample|
|no constant memory|2020_02_13__3_stable_no_constant_weights.cu|0.00120700 sec/sample|
|1st and 2nd with constants all with shared memeory with repeated mutiply|2020_02_18__05_stable_shared_memory_all_layers.cu|0.00105580 sec/sample|
|1st and 2nd with constants all with shared memeory no repeated mutiply|2020_02_20__08_stable_no_repeated_multiply_memory_all_layers.cu|0.00104740  sec/sample|
|1st and 2nd with constants all with shared memeory no repeated mutiply input from CPU|2020_02_27__09_stable_no_repeated_multiply_memory_all_layers_input_from_cpu.cu|0.00112120 sec/sample|
|batched all net in one kernel, batch size = 512|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00136102 sec/sample|
|batched all net in one kernel, batch size = 1024|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00126280 sec/sample|
|batched all net in one kernel, batch size = 2048|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00064478 sec/sample|
|batched all net in one kernel, batch size = 3072|2020_03_20__10_batched_all_net_in one_kernel.cu|0.00047949 sec/sample|


