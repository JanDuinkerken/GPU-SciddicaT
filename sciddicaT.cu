#include "util.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

// ----------------------------------------------------------------------------
// Tiled Halo Cell parameters
// ----------------------------------------------------------------------------
#define MAX_MASK_WIDTH 3
#define T_WIDTH 30
#define T_BLOCK_WIDTH (T_WIDTH + MAX_MASK_WIDTH - 1)

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) \
  ((M)[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))] = (value))
#define BUF_GET(M, rows, columns, n, i, j) \
  (M[(((n) * (rows) * (columns)) + ((i) * (columns)) + (j))])

// ----------------------------------------------------------------------------
// Inline error checking
// ----------------------------------------------------------------------------
#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char *path, int &nrows, int &ncols,
                    /*double &xllcorner, double &yllcorner, double &cellsize,*/
                    double &nodata)
{
  FILE *f;

  if ((f = fopen(path, "r")) == 0)
  {
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  // Reading the header
  char str[STRLEN];
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  ncols = atoi(str); // ncols
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nrows = atoi(str); // nrows
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // xllcorner = atof(str);  //xllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // yllcorner = atof(str);  //yllcorner
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str); // cellsize = atof(str);   //cellsize
  fscanf(f, "%s", &str);
  fscanf(f, "%s", &str);
  nodata = atof(str); // NODATA_value
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f)
  {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double *addLayer2D(int rows, int columns)
{
  double *tmp;
  gpuErrchk(cudaMallocManaged(&tmp, sizeof(double) * rows * columns));

  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop (Does not benefit of a tiled implementation)
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInitKernel(int r, int c, double *Sz,
                                              double *Sh)
{
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      double z, h;
      h = GET(Sh, c, row, col);

      if (h > 0.0)
      {
        z = GET(Sz, c, row, col);
        SET(Sz, c, row, col, z - h);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------

// This kernel does not benefit from a tiled implementation
__global__ void sciddicaTResetFlowsKernel(int r, int c, double nodata, double *Sf)
{
  int row_index = threadIdx.y + blockDim.y * blockIdx.y;
  int col_index = threadIdx.x + blockDim.x * blockIdx.x;
  int row_stride = blockDim.y * gridDim.y;
  int col_stride = blockDim.x * gridDim.x;

  for (int row = row_index + 1; row < r - 1; row += row_stride)
  {
    for (int col = col_index + 1; col < c - 1; col += col_stride)
    {
      BUF_SET(Sf, r, c, 0, row, col, 0.0);
      BUF_SET(Sf, r, c, 1, row, col, 0.0);
      BUF_SET(Sf, r, c, 2, row, col, 0.0);
      BUF_SET(Sf, r, c, 3, row, col, 0.0);
    }
  }
}

// This kernel benefits from a tiled implementation
__global__ void sciddicaTFlowsComputationHaloKernel(int r, int c, double nodata,
                                                    int *Xi, int *Xj, double *Sz, double *Sh,
                                                    double *Sf, double p_r, double p_epsilon)
{
  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  __shared__ double Sz_ds[T_BLOCK_WIDTH][T_BLOCK_WIDTH];
  __shared__ double Sh_ds[T_BLOCK_WIDTH][T_BLOCK_WIDTH];

  int col_index = threadIdx.x + T_WIDTH * blockIdx.x;
  int row_index = threadIdx.y + T_WIDTH * blockIdx.y;
  long col_halo = col_index - MAX_MASK_WIDTH / 2;
  long row_halo = row_index - MAX_MASK_WIDTH / 2;

  if ((row_halo >= 0) && (row_halo < r) && (col_halo >= 0) && (col_halo < c))
  {
    Sz_ds[threadIdx.y][threadIdx.x] = GET(Sz, c, row_halo, col_halo);
    Sh_ds[threadIdx.y][threadIdx.x] = GET(Sh, c, row_halo, col_halo);
  }
  else
  {
    Sz_ds[threadIdx.y][threadIdx.x] = nodata;
    Sh_ds[threadIdx.y][threadIdx.x] = nodata;
  }
  __syncthreads();

  if (row_index >= 1 && row_index < r - 1 && col_index >= 1 && col_index < c - 1)
  {
    int index_i = threadIdx.y + MAX_MASK_WIDTH/2;
    int index_j = threadIdx.x + MAX_MASK_WIDTH/2;
    if (index_i >= 1 && index_i <= T_WIDTH && index_j >= 1 && index_j <= T_WIDTH)
    {
      m = Sh_ds[index_i][index_j] - p_epsilon;
      u[0] = Sz_ds[index_i][index_j] + p_epsilon;

      z = Sz_ds[index_i + Xi[1]][index_j + Xj[1]];
      h = Sh_ds[index_i + Xi[1]][index_j + Xj[1]];
      u[1] = z + h;

      z = Sz_ds[index_i + Xi[2]][index_j + Xj[2]];
      h = Sh_ds[index_i + Xi[2]][index_j + Xj[2]];
      u[2] = z + h;

      z = Sz_ds[index_i + Xi[3]][index_j + Xj[3]];
      h = Sh_ds[index_i + Xi[3]][index_j + Xj[3]];
      u[3] = z + h;

      z = Sz_ds[index_i + Xi[4]][index_j + Xj[4]];
      h = Sh_ds[index_i + Xi[4]][index_j + Xj[4]];
      u[4] = z + h;

      do
      {
        again = false;
        average = m;
        cells_count = 0;

        for (n = 0; n < 5; n++)
          if (!eliminated_cells[n])
          {
            average += u[n];
            cells_count++;
          }

        if (cells_count != 0)
        {
          average /= cells_count;
        }

        for (n = 0; n < 5; n++)
        {
          if ((average <= u[n]) && (!eliminated_cells[n]))
          {
            eliminated_cells[n] = true;
            again = true;
          }
        }
      } while (again);

      if (!eliminated_cells[1])
      {
        BUF_SET(Sf, r, c, 0, row_index, col_index, (average - u[1]) * p_r);
      }
      if (!eliminated_cells[2])
      {
        BUF_SET(Sf, r, c, 1, row_index, col_index, (average - u[2]) * p_r);
      }
      if (!eliminated_cells[3])
      {
        BUF_SET(Sf, r, c, 2, row_index, col_index, (average - u[3]) * p_r);
      }
      if (!eliminated_cells[4])
      {
        BUF_SET(Sf, r, c, 3, row_index, col_index, (average - u[4]) * p_r);
      }
    }
  }
}

// This kernel benefits from a tiled implementation
__global__ void sciddicaTWidthUpdateHaloKernel(int r, int c, double nodata, int *Xi, int *Xj, double *Sz, double *Sh, double *Sf)
{
  int row_index = threadIdx.y + T_WIDTH * blockIdx.y;
  int col_index = threadIdx.x + T_WIDTH * blockIdx.x;
  long row_halo = row_index - MAX_MASK_WIDTH / 2;
  long col_halo = col_index - MAX_MASK_WIDTH / 2;

  double h_next = 0.0;

  __shared__ double Sf_ds[T_BLOCK_WIDTH * ADJACENT_CELLS][T_BLOCK_WIDTH];

  if ((col_halo >= 0) && (col_halo < c) && (row_halo >= 0) && (row_halo < r))
  {
    Sf_ds[threadIdx.y][threadIdx.x] = BUF_GET(Sf, r, c, 0, row_halo, col_halo);
    Sf_ds[threadIdx.y + T_BLOCK_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, 1, row_halo, col_halo);
    Sf_ds[threadIdx.y + 2 * T_BLOCK_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, 2, row_halo, col_halo);
    Sf_ds[threadIdx.y + 3 * T_BLOCK_WIDTH][threadIdx.x] = BUF_GET(Sf, r, c, 3, row_halo, col_halo);
  }
  else
  {
    Sf_ds[threadIdx.y][threadIdx.x] = nodata;
  }
  __syncthreads();

  if (row_index >= 1 && col_index >= 1 && row_index < r - 1 && col_index < c - 1)
  {
    int index_i = threadIdx.y + MAX_MASK_WIDTH / 2;
    int index_j = threadIdx.x + MAX_MASK_WIDTH / 2;
    if (index_i >= 1 && index_i <= T_WIDTH && index_j >= 1 && index_j <= T_WIDTH)
    {
      h_next = GET(Sh, c, row_index, col_index);
      h_next += Sf_ds[index_i + Xi[1] + (T_BLOCK_WIDTH * 3)][index_j + Xj[1]] - Sf_ds[index_i][index_j];
      h_next += Sf_ds[index_i + Xi[2] + (T_BLOCK_WIDTH * 2)][index_j + Xj[2]] - Sf_ds[index_i + T_BLOCK_WIDTH][index_j];
      h_next += Sf_ds[index_i + Xi[3] + T_BLOCK_WIDTH][index_j + Xj[3]] - Sf_ds[index_i + (T_BLOCK_WIDTH * 2)][index_j];
      h_next += Sf_ds[index_i + Xi[4]][index_j + Xj[4]] - Sf_ds[index_i + (T_BLOCK_WIDTH * 3)][index_j];

      SET(Sh, c, row_index, col_index, h_next);
    }
  }
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows; // r: grid rows
  int c = cols; // c: grid columns
  double *Sz;   // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;   // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;   // Sf: 4 substates containing the flows towards the 4 neighs

  int *Xi;
  int *Xj;

  gpuErrchk(cudaMallocManaged(&Xi, sizeof(int) * 5));
  gpuErrchk(cudaMallocManaged(&Xj, sizeof(int) * 5));

  // Xj: von Neuman neighborhood row coordinates (see below)
  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;
  // Xj: von Neuman neighborhood col coordinates (see below)
  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;

  double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); // steps: simulation steps

  dim3 tiled_block_size(T_BLOCK_WIDTH, T_BLOCK_WIDTH, 1); // == T_BUFF_SIZE
  dim3 tiled_grid_size(ceil(rows / T_WIDTH), ceil(cols / T_WIDTH), 1);

  // Not all kernels are going to use a tiled implementation so we keep the normal grid and block size variables
  int n = rows * cols;
  int dim_x = 32;
  int dim_y = 32;
  dim3 block_size(dim_x, dim_y, 1);
  dim3 grid_size(ceil(sqrt(n / (dim_x * dim_y))), ceil(sqrt(n / (dim_x * dim_y))), 1);

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the
  //   neighborhood flow_index in   [0,1,2,3]: outgoing flow indices in Sf from
  //   cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //

  Sz = addLayer2D(r, c); // Allocates the Sz substate grid
  Sh = addLayer2D(r, c); // Allocates the Sh substate grid
  Sf = addLayer2D(ADJACENT_CELLS * r,
                  c); // Allocates the Sf substates grid,
                      //   having one layer for each adjacent cell

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid
  // (cellular space)
  sciddicaTSimulationInitKernel<<<grid_size, block_size>>>(r, c, Sz, Sh);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
    // Apply the resetFlow kernel to the whole domain
    sciddicaTResetFlowsKernel<<<grid_size, block_size>>>(r, c, nodata, Sf);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // Apply the FlowComputation kernel to the whole domain
    sciddicaTFlowsComputationHaloKernel<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // Apply the WidthUpdate mass balance kernel to the whole domain
    sciddicaTWidthUpdateHaloKernel<<<tiled_grid_size, tiled_block_size>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]); // Save Sh to file

  printf("Releasing memory...\n");
  gpuErrchk(cudaFree(Sz));
  gpuErrchk(cudaFree(Sh));
  gpuErrchk(cudaFree(Sf));

  return 0;
}
