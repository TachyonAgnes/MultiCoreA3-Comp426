#include "Render_CUDA.cuh"
#include "Cell_CUDA.cuh"
#include <iostream>

Cell *hostCurrentCells = nullptr;

void Render::InitHostCells(int size) {
	if (hostCurrentCells == nullptr) {
		hostCurrentCells = new Cell[size];
	}
}

void Render::ActualDisplay() {
	// clear buffer
	glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_POINTS);
	// access currentCellsBuffer by row and col 
	Grid &grid = Grid::GetInstance();
	int size = displaySize.width * displaySize.height;
	Cell *d_cells = grid.GetCurrentCellsBuffer();

	cudaMemcpy(hostCurrentCells, d_cells, size * sizeof(Cell), cudaMemcpyDeviceToHost);

	Cell *cells = grid.GetCurrentCellsBuffer();
	for (int row = 0; row < displaySize.height; ++row) {
		for (int col = 0; col < displaySize.width; ++col) {
			int index = row * displaySize.width + col;
			Cell &cell = hostCurrentCells[index];
			switch (cell.GetType()) {
				case CellType::CANCER:
					glColor3f(colorRed.r, colorRed.g, colorRed.b);
					break;
				case CellType::HEALTHY:
					glColor3f(colorGreen.r, colorGreen.g, colorGreen.b);
					break;
				case CellType::MEDICINE:
					glColor3f(colorYellow.r, colorYellow.g, colorYellow.b);
					break;
				default:
					glColor3f(0, 0, 0);
					break;
			}
			glVertex2f((float)cell.GetX(), (float)cell.GetY());
		}
	}
	glEnd();

	// flush buffer
	glutSwapBuffers();
}

void Render::Reshape(int w, int h) {
	glutReshapeWindow(GetInstance().displaySize.width, GetInstance().displaySize.height);
}

void Render::Cleanup() {
	if (hostCurrentCells != nullptr) {
		delete[] hostCurrentCells;
		hostCurrentCells = nullptr;
	}
}