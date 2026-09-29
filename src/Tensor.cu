#include <iostream>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <memory>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

enum class Device {
    CPU,
    GPU
};

void checkCuda(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(err)
        );
    }
}

class Storage {
    float* buffer = nullptr;
    std::size_t count = 0;
    Device location = Device::CPU;

public:
    Storage() {}

    Storage(std::size_t n, Device where)
        : count(n), location(where) {

        // 即使 count 为 0，也先检查 device 是否有效。
        if (location != Device::CPU && location != Device::GPU) {
            throw std::invalid_argument("device must be CPU or GPU");
        }

        if (count == 0) {
            return;
        }

        if (location == Device::CPU) {
            buffer = new float[count];
        } else {
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&buffer),
                    count * sizeof(float)
                ),
                "cudaMalloc"
            );
        }
    }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    ~Storage() noexcept {
        if (buffer == nullptr) {
            return;
        }

        if (location == Device::CPU) {
            delete[] buffer;
        } else {
            cudaError_t err = cudaFree(buffer);
            if (err != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "cudaFree failed: %s\n",
                    cudaGetErrorString(err)
                );
            }
        }
    }

    float* data() { return buffer; }
    const float* data() const { return buffer; }

    std::size_t capacity() const { return count; }
    Device device() const { return location; }
};

class Tensor {
    std::vector<std::size_t> shape;
    std::size_t n = 0;
    std::shared_ptr<Storage> storage;

public:
    Tensor() {
        storage = std::make_shared<Storage>();
    }

    Tensor(std::vector<std::size_t> v, Device where) {
        shape = v;

        if (!v.empty()) {
            n = 1;
        }

        for (std::vector<std::size_t>::iterator it = v.begin();
             it != v.end(); ++it) {
            n *= *it;
        }

        storage = std::make_shared<Storage>(n, where);
    }

    float* data() {
        return storage->data();
    }

    const float* data() const {
        return storage->data();
    }

    std::size_t numel() const {
        return n;
    }

    Device device() const {
        return storage->device();
    }

    Tensor clone() const {
        Tensor t(shape, device());

        if (device() == Device::CPU) {
            for (std::size_t i = 0; i < n; ++i) {
                t.data()[i] = data()[i];
            }
        } else if (n > 0) {
            checkCuda(
                cudaMemcpy(
                    t.data(), data(), n * sizeof(float),
                    cudaMemcpyDeviceToDevice
                ),
                "Tensor::clone cudaMemcpy"
            );
        }

        return t;
    }

    Tensor cpu() const {
        if (device() == Device::CPU) {
            Tensor t;
            t.shape = shape;
            t.storage = storage;
            t.n = n;
            return t;
        } else {
            Tensor t(shape, Device::CPU);

            if (n > 0) {
                checkCuda(
                    cudaMemcpy(
                        t.data(), data(), n * sizeof(float),
                        cudaMemcpyDeviceToHost
                    ),
                    "Tensor::cpu cudaMemcpy"
                );
            }

            return t;
        }
    }

    Tensor gpu() const {
        if (device() == Device::GPU) {
            Tensor t;
            t.shape = shape;
            t.storage = storage;
            t.n = n;
            return t;
        } else {
            Tensor t(shape, Device::GPU);

            if (n > 0) {
                checkCuda(
                    cudaMemcpy(
                        t.data(), data(), n * sizeof(float),
                        cudaMemcpyHostToDevice
                    ),
                    "Tensor::gpu cudaMemcpy"
                );
            }

            return t;
        }
    }
};