# try to make cuda code that faster then tensorflow for lenet  

Here I made LeNet that recognizes MNIST. First I made it in Tensorflow then in CUDA (only inference). I tried to make optimizations in CUDA code to make it faster then Tensorflow inference.  Here is final results:  
Tensorflow 1.4 vs CUDA on GPU GTX760:  
![tf vs cuda](/times_semilogy_tf_1_4_vs_cuda_gtx760__2020_03_23.png)  
Tensorflow 1.4 vs Tensorflow 1.15 on GPU Tesla K80:
![tf 1.4 vs tf 1.5](/times_semilogy_tf_1_4_vs_tf_1_15_k80__2020_03_23.png)  


