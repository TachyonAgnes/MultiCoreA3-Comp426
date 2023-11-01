#include "Logic_CUDA.cuh"


// Callback function to handle mouse clicks for injecting medicine into cells.
void Logic::MedicineInjection(int button, int state, int x, int y) {
	if (button != GLUT_LEFT_BUTTON || state != GLUT_DOWN) return;
	// Get the singleton instances of the grid and render classes.
	Grid &gridInstance = Grid::GetInstance();
	Render &renderInstance = Render::GetInstance();
	DisplaySize displaySize = renderInstance.displaySize;
	Cell *currentCellsBuffer = gridInstance.GetCurrentCellsBuffer();
	Cell *nextCellsBuffer = gridInstance.GetNextCellsBuffer();
	
	dim3 blockDim(16,16);
	dim3 gridDim((displaySize.width + blockDim.x - 1) / blockDim.x, (displaySize.height + blockDim.y - 1) / blockDim.y);

	unsigned short radius = 1;
	MedicineInjectionKernel<<<gridDim, blockDim>>>(currentCellsBuffer, nextCellsBuffer, displaySize.width, displaySize.height, x, y, 1);
	cudaDeviceSynchronize();
	cudaMemcpy(currentCellsBuffer, nextCellsBuffer, displaySize.width * displaySize.height * sizeof(Cell), cudaMemcpyDeviceToDevice);
};

__global__ void MedicineInjectionKernel(Cell *currentCells, Cell *nextCells, int width, int height, int x, int y, unsigned short radius) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if(col>= width || row>=height) return;

	int index = row * width + col;
	Cell *pointedCell = &currentCells[index];

	if (pointedCell->GetType() == CellType::CANCER) return;

	int winY = height - y;

	if (abs(pointedCell->GetX() - x) <= radius && abs(pointedCell->GetY() - winY) <= radius) {
		if (pointedCell->GetType() != CellType::CANCER) {
			nextCells[index].SetType(CellType::MEDICINE);
		}
	}
}

void Logic::UpdateAllCell() {
	cudaDeviceSynchronize();
	DisplaySize displaySize = Render::GetInstance().displaySize;
	int height = displaySize.height;
	int width = displaySize.width;
	Cell *currCellsBuffer = Grid::GetInstance().GetCurrentCellsBuffer();
	Cell *nextCellsBuffer = Grid::GetInstance().GetNextCellsBuffer();

	cudaSetDevice(0);

	// define the block size and grid size
	dim3 blockSize(16,16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// call the kernel function
	UpdateAllCellKernel<<<gridSize, blockSize>>>(currCellsBuffer, nextCellsBuffer, width, height);
	cudaDeviceSynchronize();

	// update buffer
	cudaMemcpy(currCellsBuffer, nextCellsBuffer, width * height * sizeof(Cell), cudaMemcpyDeviceToDevice);
}

__global__ void UpdateAllCellKernel(Cell *currentCellsBuffer, Cell *nextCellsBuffer, int width, int height) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= width || row >= height) return;
	int index = row * width + col;
	Cell *currentCell = &currentCellsBuffer[index];
	Cell *nextCell = &nextCellsBuffer[index];
	UpdateCell(currentCell, nextCell, currentCellsBuffer, nextCellsBuffer, width, height);
}

__device__ void UpdateCell(Cell *currentCell, Cell *nextCell, Cell *currentCellsBuffer, Cell *nextCellsBuffer, int width, int height) {
	// Get the type and position of the current cell.
	CellType cellType = currentCell->GetType();
	int posX = currentCell->GetX();
	int posY = currentCell->GetY();
	unsigned short radius = 1;

	unsigned short minHeight = (0 > (posY - radius)) ? 0 : posY - radius;
	unsigned short maxHeight = (height < (posY + radius + 1)) ? height : posY + radius + 1;
	unsigned short minWidth = (0 > (posX - radius)) ? 0 : posX - radius;
	unsigned short maxWidth = (width < (posX + radius + 1)) ? width : posX + radius + 1;

	int cancerNeighborCount = 0;
	int medicineNeighborCount = 0;

	// Loop through neighboring cells to determine the next state of the current cell.
	bool shouldBreak = false;
	for (unsigned short newRow = minHeight; newRow < maxHeight && !shouldBreak; newRow++) {
		for (unsigned short newCol = minWidth; newCol < maxWidth && !shouldBreak; newCol++) {
			if (newRow == posY && newCol == posX) continue;
			int index = newRow * width + newCol;
			Cell &currNeighborCell = currentCellsBuffer[index];
			Cell &nextNeighborCell = nextCellsBuffer[index];

			switch (cellType) {
			case CellType::MEDICINE:
				if (currNeighborCell.GetType() == CellType::HEALTHY) {
					nextNeighborCell.SetType(CellType::MEDICINE);
				}
				break;
			case CellType::HEALTHY:
				if (currNeighborCell.GetType() == CellType::CANCER) {
					cancerNeighborCount++;
					if (cancerNeighborCount >= 6) {
						nextCell->SetType(CellType::CANCER);
						shouldBreak = true;
					}
				}
				break;
			case CellType::CANCER:
				if (currNeighborCell.GetType() == CellType::MEDICINE) {
					medicineNeighborCount++;
					if (medicineNeighborCount >= 6) {
						nextCell->SetType(CellType::HEALTHY);
						for (unsigned short i = minHeight; i < maxHeight; i++) {
							for (unsigned short j = minWidth; j < maxWidth; j++) {
								if (i == posY && j == posX) continue;
								int indexN = i * width + j;
								Cell &surroundingCell = nextCellsBuffer[indexN];
								if (surroundingCell.GetType() == CellType::MEDICINE) {
									surroundingCell.SetType(CellType::HEALTHY);
								}
							}
						}	
						shouldBreak = true;
					}
				}
				break;
			default:
				break;
			}
		}
	}
}
