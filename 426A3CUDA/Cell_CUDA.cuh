#pragma once
#include <cuda_runtime.h>
#include "Render_CUDA.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cstring> 

enum CellType {
    HEALTHY,
    CANCER,
    MEDICINE
};

class Cell {
private:
    CellType type;
    int x, y;
public:
    __host__ __device__ Cell(CellType type, int x, int y): type(type), x(x), y(y) {}
    __host__ __device__ Cell(): type(CellType::HEALTHY), x(0), y(0) {}

    __host__ __device__ void SetType(CellType newType) { type = newType; }
    __host__ __device__ CellType GetType() const { return type; }

    __host__ __device__ void SetPosition(int newX, int newY) {
        x = newX; y = newY;
    }

    __host__ __device__ int GetX() const { return x; }
    __host__ __device__ int GetY() const { return y; }
};

class Grid {
private:
    static Grid *instance;
    int width, height;
    Cell *currentCellsBuffer; // for rendering
    Cell *nextCellsBuffer; // for data update

    Grid(int width, int height): width(width), height(height) {
        size_t size = width * height * sizeof(Cell);
        cudaMalloc(&currentCellsBuffer, size);
        cudaMalloc(&nextCellsBuffer, size);
    }

public:
    // singleton pattern
    static Grid &GetInstance() {
        if (instance == nullptr) {
            int width = Render::GetInstance().displaySize.width;
            int height = Render::GetInstance().displaySize.height;
            instance = new Grid(width, height);
        }
        return *instance;
    }

    Grid(const Grid &) = delete;
    Grid &operator=(const Grid &) = delete;

    ~Grid() {
        cudaFree(currentCellsBuffer);
        cudaFree(nextCellsBuffer);
    }
    // Get methods
    __host__ __device__ Cell *GetCurrentCellsBuffer() const {
        return currentCellsBuffer;
    }

    __host__ __device__ Cell *GetNextCellsBuffer() const {
        return nextCellsBuffer;
    }

    // Set methods
    __host__ void SetCurrentCellsBuffer_D2D(Cell *buffer) {
        if (buffer != nullptr) {
            cudaMemcpy(currentCellsBuffer, buffer, width * height * sizeof(Cell), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
        }
    }

    __host__ void SetNextCellsBuffer_D2D(Cell *buffer) {
        if (buffer != nullptr) {
            cudaMemcpy(nextCellsBuffer, buffer, width * height * sizeof(Cell), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        }   
    }
};

__global__ void InitKernel(Cell *cells, int width, int height, unsigned long long seed);
