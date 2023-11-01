#include "Logic_CUDA.cuh"
#include "Cell_CUDA.cuh"
#include "Render_CUDA.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <ctime>
#include <thread>

// Initializes the grid of cells.
void GridInit(int height, int width) {
	Grid &grid = Grid::GetInstance();
	Cell *currentCellsBuffer = grid.GetCurrentCellsBuffer();

	dim3 block_dim(16, 16);
	dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);


	unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
	InitKernel<<<grid_dim, block_dim>>> (currentCellsBuffer, width, height, seed);
	cudaDeviceSynchronize();
	Grid::GetInstance().SetNextCellsBuffer_D2D(currentCellsBuffer);
}

/// ENVIRONMENT INITIALIZATION
void InitGLUT(int *argc, char **argv, DisplaySize displaySize) {
	// Setting up window size, display mode and creating window for the simulation.
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(Render::GetInstance().displaySize.width, Render::GetInstance().displaySize.height);
	glutCreateWindow("2D Cell Growth Simulation");
}

void InitGLEW() {
	// Throws runtime error if GLEW fails to initialize.
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		throw std::runtime_error("Failed to initialize GLEW");
	}
}

// Sets up Orthographic Projection for 2D rendering.
void InitOrthoProjection(DisplaySize displaySize) {
	// change to orthographic projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, displaySize.width, 0, displaySize.height);
	glMatrixMode(GL_MODELVIEW);
}

// Registers Callback functions for reshaping and displaying.
void RegisterCallbacks() {
	// register callbacks
	glutReshapeFunc(Render::Reshape);
	glutDisplayFunc(Render::Display);
}

/// THREAD FUNCTIONS
 //Contains the main loop of the program where cells are updated every 33 milliseconds (1s/30).
void MainLoop(int ms) {
	while (true) {
		Logic::GetInstance().UpdateAllCell();
		std::cout << "__MainLoopOnUpdate__" << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}
}

void timer(int ms) {
	glutPostRedisplay();  // redraw window
	glutTimerFunc(ms, timer, 0);  // call timer function after 33 milliseconds
}

/// MAIN FUNCTION
int main(int argc, char **argv) {
	try {
		// Render::init variables
		DisplaySize displaySize = Render::GetInstance().displaySize;
		Color colorGreen = Render::GetInstance().colorGreen;

		// Initialize GLUT, GLEW, and register callbacks
		InitGLUT(&argc, argv, displaySize);
		InitGLEW();
		InitOrthoProjection(displaySize);
		glViewport(0, 0, displaySize.width, displaySize.height);
		glClearColor(0, 0, 0, 1.0f);
		Render::GetInstance().InitHostCells(displaySize.width * displaySize.height);
		// Register Callbacks
		RegisterCallbacks();
		GridInit(displaySize.height, displaySize.width);
		glutMouseFunc(Logic::MedicineInjection);
		int latency = 33;

		/// Main Loop
		std::thread t1(MainLoop, latency);
		t1.detach();

		/// Render loop
		glutTimerFunc(latency, timer, 0);
		glutMainLoop();
		Render::GetInstance().Cleanup();

	} catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
	return 0;
}
