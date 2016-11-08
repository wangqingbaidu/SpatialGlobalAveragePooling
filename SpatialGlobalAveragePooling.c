#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialGlobalAveragePooling.c"
#else

void THNN_(SpatialGlobalAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
	printf("SpatialGlobalAveragePooling in CPU is not implemented!");
	exit(-1);
}

void THNN_(SpatialGlobalAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput)
{
	printf("SpatialGlobalAveragePooling in CPU is not implemented!");
	exit(-1);
}

#endif
