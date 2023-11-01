#include "Cell_CUDA.cuh"

Grid *Grid::instance = nullptr;

__global__ void InitKernel(Cell *cells, int width, int height, unsigned long long seed) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int index = row * width + col;
	if (row < height && col < width) {
		cells[index].SetType(CellType::HEALTHY);
		cells[index].SetPosition(col, row);

		curandState state;
		curand_init(seed, index, 0, &state);

		float randValue = curand_uniform(&state);

		if (randValue < 0.25f) {
			cells[index].SetType(CellType::CANCER);
		}
	}
}
