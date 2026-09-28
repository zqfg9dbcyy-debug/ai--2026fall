#include<iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include<string>
#include<vector>

class Tensor{
    std::string device;
    float* data;
    std::size_t n=1;
    std::vector<size_t> shape;
    public:
    Tensor(std::vector<size_t> a,std::string d){
        shape=a;
        device=d;
        for(std::vector<size_t>::iterator it=shape.begin();it!=shape.end();++it){
            n*=*it;
        }
        if(device=="GPU"){
            cudaMalloc(&data,n*sizeof(float));
        }
        else{
            cudaMallocHost(&data,n*sizeof(float));
        }
    }
};