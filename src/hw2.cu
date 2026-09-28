#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

class Tensor{
    public:
    std::string device;
    float *data = nullptr;
    std::size_t n = 0;
    std::vector<size_t> shape;
    Tensor(std::vector<size_t> a, std::string d){
        shape = a;
        device = d;
        n = shape.empty() ? 0 : 1;
        if (n == 1){
            for (std::vector<size_t>::iterator it = shape.begin(); it != shape.end(); ++it){
                n *= *it;
            }
            if (device == "GPU"){
                cudaMalloc(&data, n * sizeof(float));
            }
            else if (device == "CPU"){
                cudaMallocHost(&data, n * sizeof(float));
            }
            else{
                throw std::invalid_argument("device must be CPU or GPU");
            }
        }
    }
    Tensor() : shape(), device("CPU"), n(0){}
    Tensor(const Tensor &t){
        shape = t.shape;
        device = t.device;
        n = t.n;
        if (n > 0){
            if (device == "GPU"){
                cudaMalloc(&data, n * sizeof(float));
                cudaMemcpy(data, t.data, n * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            else{
                cudaMallocHost(&data, n * sizeof(float));
                cudaMemcpy(data, t.data, n * sizeof(float), cudaMemcpyHostToHost);
            }
        }
    }
    Tensor &operator=(const Tensor &t){
        if (this == &t){
            return *this;
        }
        if (data != nullptr) {
            if (device == "GPU") {
                cudaFree(data);
            } else {
                cudaFreeHost(data);
            }
        }
        data = nullptr;
        shape = t.shape;
        device = t.device;
        n = t.n;
        if (n > 0){
            if (device == "GPU"){
                cudaMalloc(&data, n * sizeof(float));
                cudaMemcpy(data, (t.data), n * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            else{
                cudaMallocHost(&data, n * sizeof(float));
                cudaMemcpy(data, (t.data), n * sizeof(float), cudaMemcpyHostToHost);
            }
        }
        return *this;
    }
    Tensor gpu(){
        Tensor t;
        t.shape = shape;
        t.n = n;
        t.device = "GPU";
        if(n==0) return t;
        cudaMalloc(&t.data, n * sizeof(float));
        if (device == "GPU"){
            cudaMemcpy((t.data), data, n * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else{
            cudaMemcpy((t.data), data, n * sizeof(float), cudaMemcpyHostToDevice);
        }
        return t;
    }
    Tensor cpu(){
        Tensor t;
        t.shape = shape;
        t.n = n;
        t.device = "CPU";
        if(n==0) return t;
        cudaMallocHost(&t.data, n * sizeof(float));
        if (device == "GPU"){
            cudaMemcpy((t.data), data, n * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else{
            cudaMemcpy((t.data), data, n * sizeof(float), cudaMemcpyHostToHost);
        }
        return t;
    }
    ~Tensor(){
        if (data != nullptr) {
            if (device == "GPU") {
                cudaFree(data);
            }   
            else {
                cudaFreeHost(data);
            }
        }
    }
};

__global__ void ReLU(float *input,float *output,int N){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<N){
        output[i]=input[i]>0?input[i]:0;
    }
}

__global__ void ReLU_Forward(float *input,float *grad_output,float *grad_input,int N){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<N){
        grad_output[i]=input[i]>0?grad_input[i]:0;
    }
}

__global__ void sigmoid(float *input,float *output,int N){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<N){
        if(input[i]>=0){
            output[i]=1/(1+expf(-input[i]));
        }
        else{
            float e=exp(-input[i]);
            output[i]=1/(1+e);
        }
    }
}

__global__ void sigmoid_Forward(float *input,float *grad_output,float *grad_input,int N){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<N){
        float x;
        if(input[i]>=0){
            x=1/(1+expf(-input[i]));
        }
        else{
            float e=exp(-input[i]);
            x=1/(1+e);
        }
        grad_output[i]=grad_input[i]*(1-x)*x;
    }
}

int main(){
    
}