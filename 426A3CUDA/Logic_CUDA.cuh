//#pragma once
#include "Cell_CUDA.cuh"
#include "Render_CUDA.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

class Logic {
	private: 
		Logic() {};
	public:
		// singleton pattern
		static Logic &GetInstance() {
			static Logic instance;
			return instance;
		}
		Logic(const Logic &) = delete;
		Logic &operator=(const Logic &) = delete;

		// operate the cells
		static void MedicineInjection(int button, int state, int x, int y);
		void UpdateAllCell();
};

__global__ void UpdateAllCellKernel(Cell *currentCellsBuffer, Cell *nextCellsBuffer, int width, int height);

__device__ void UpdateCell(Cell *currentCell, Cell *nextCell, Cell *currentCellsBuffer, Cell *nextCellsBuffer, int width, int height);

__global__ void MedicineInjectionKernel(Cell *currentCells, Cell *nextCells, int width, int height, int x, int y, unsigned short radius);