#include "THCUNN.h"
#include "common.h"

template <typename Dtype>
__global__ void GlobalAvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {    
    const int pool_size = height * width;
    Dtype aveval = 0;
    const Dtype* const bottom_slice = bottom_data + index * height * width;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
   top_data[index] = aveval / pool_size;
  }
}


void THNN_CudaSpatialGlobalAveragePooling_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  long nInputCols, nInputRows, nInputPlane, batchSize;
  long nOutputCols = 1, nOutputRows = 1;

  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }
 
  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nOutputRows, nOutputCols);

  float* output_data = THCudaTensor_data(state, output);

  int count = THCudaTensor_nElement(state, output);

  GlobalAvePoolForward<float>
	  <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
		count, input_data, batchSize, nInputPlane, nInputRows, nInputCols, output_data);
  
  THCudaCheck(cudaGetLastError());

  if(input->nDimension == 3)
    THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows, nOutputCols);

  THCudaTensor_free(state, input);

}

template <typename Dtype>
__global__ void GlobalAvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height, const int width, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int pool_size = height * width;
    const Dtype* const top_diff_slice = top_diff + index / pool_size;
    Dtype gradient = top_diff_slice[0] / pool_size;
    bottom_diff[index] = gradient;
  }
}

void THNN_CudaSpatialGlobalAveragePooling_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  input = THCudaTensor_newContiguous(state, input);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  long nInputCols, nInputRows, nInputPlane, batchSize;
//  long nOutputCols = 1, nOutputRows = 1;
  
  if (input->nDimension == 3) {
    nInputCols = input->size[2];
    nInputRows = input->size[1];
    nInputPlane = input->size[0];
    batchSize = 1;
  }
  else
  {
    nInputCols = input->size[3];
    nInputRows = input->size[2];
    nInputPlane = input->size[1];
    batchSize = input->size[0];
  }

  THCudaTensor_resizeAs(state, gradInput, input);

  int count = THCudaTensor_nElement(state, input);
  
    GlobalAvePoolBackward<float>
      <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
        (count,
        THCudaTensor_data(state, gradOutput),
        batchSize, nInputPlane, nInputRows, nInputCols,
        THCudaTensor_data(state, gradInput));
  
  THCudaCheck(cudaGetLastError());

  // clean
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
}

