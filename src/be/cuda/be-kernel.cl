__global__ void BackgroundExtraction(uint8_t *frame, float *bg, uint8_t *fg,
                                     uint32_t width, uint32_t height,
                                     uint32_t channel, uint8_t threshold,
                                     float alpha) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > width * height * channel) {
    return;
  }

  uint8_t diff;
  if (frame[tid] > bg[tid]) {
    diff = frame[tid] - bg[tid];
  } else {
    diff = bg[tid] - frame[tid];
  }

  if (diff > threshold) {
    fg[tid] = frame[tid];
  } else {
    fg[tid] = 0;
  }

  bg[tid] = bg[tid] * (1 - alpha) + frame[tid] * alpha;
}

