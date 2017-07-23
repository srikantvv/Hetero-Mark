/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Carter McCardwell (carter@mccardwell.net, cmccardw@ece.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 *  with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/
 * or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimers. Redistributions in binary
 *   form must reproduce the above copyright notice, this list of conditions and
 *   the following disclaimers in the documentation and/or other materials
 *   provided with the distribution. Neither the names of NUCAR, Northeastern
 *   University, nor the names of its contributors may be used to endorse or
 *   promote products derived from this Software without specific prior written
 *   permission.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *   DEALINGS WITH THE SOFTWARE.
 */

#include <cstring>
#include <memory>
#include <string>
#include "src/aes/cuda/aes_cuda_benchmark.h"

void AesCudaBenchmark::Initialize() {
  AesBenchmark::Initialize();
  ExpandKey();

  cudaMallocManaged((void **)&dev_ciphertext_, text_length_ * sizeof(uint8_t));
  cudaMallocManaged((void **)&dev_key_, kExpandedKeyLengthInBytes);
  cudaMallocManaged((void **)&dev_s_, 256 * sizeof(uint8_t));

  memcpy(dev_ciphertext_, ciphertext_, text_length_);
  memcpy(dev_key_, expanded_key_, kExpandedKeyLengthInBytes);
  memcpy(dev_s_, s, 256 * sizeof(uint8_t));
}

__device__ void AddRoundKeyGpu(uint8_t *state, uint32_t *exp_key, int offset) {
  uint8_t *key_bytes = (uint8_t *)(exp_key) + 16 * offset;
  state[0] ^= key_bytes[3];
  state[1] ^= key_bytes[2];
  state[2] ^= key_bytes[1];
  state[3] ^= key_bytes[0];
  state[4] ^= key_bytes[7];
  state[5] ^= key_bytes[6];
  state[6] ^= key_bytes[5];
  state[7] ^= key_bytes[4];
  state[8] ^= key_bytes[11];
  state[9] ^= key_bytes[10];
  state[10] ^= key_bytes[9];
  state[11] ^= key_bytes[8];
  state[12] ^= key_bytes[15];
  state[13] ^= key_bytes[14];
  state[14] ^= key_bytes[13];
  state[15] ^= key_bytes[12];
}

__device__ void SubBytesGpu(uint8_t *state, uint8_t *s) {
  for (int i = 0; i < 16; i++) {
    state[i] = s[state[i]];
  }
}

__device__ void ShiftRowsGpu(uint8_t *state) {
  uint8_t new_state[16];
  new_state[0] = state[0];
  new_state[1] = state[5];
  new_state[2] = state[10];
  new_state[3] = state[15];
  new_state[4] = state[4];
  new_state[5] = state[9];
  new_state[6] = state[14];
  new_state[7] = state[3];
  new_state[8] = state[8];
  new_state[9] = state[13];
  new_state[10] = state[2];
  new_state[11] = state[7];
  new_state[12] = state[12];
  new_state[13] = state[1];
  new_state[14] = state[6];
  new_state[15] = state[11];

  state[0] = new_state[0];
  state[1] = new_state[1];
  state[2] = new_state[2];
  state[3] = new_state[3];
  state[4] = new_state[4];
  state[5] = new_state[5];
  state[6] = new_state[6];
  state[7] = new_state[7];
  state[8] = new_state[8];
  state[9] = new_state[9];
  state[10] = new_state[10];
  state[11] = new_state[11];
  state[12] = new_state[12];
  state[13] = new_state[13];
  state[14] = new_state[14];
  state[15] = new_state[15];
}

__device__ void MixColumnsGpu(uint8_t *state) {
  for (int i = 0; i < 4; i++) {
    uint8_t *word = state + 4 * i;
    uint8_t a[4];
    uint8_t b[4];
    uint8_t high_bit;
    for (int i = 0; i < 4; i++) {
      a[i] = word[i];
      high_bit = word[i] & 0x80;
      b[i] = word[i] << 1;
      if (high_bit == 0x80) {
        b[i] ^= 0x1b;
      }
    }
    word[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
    word[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
    word[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
    word[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];
  }
}

__global__ void aes_cuda(uint8_t *input, uint32_t *expanded_key, uint8_t *s) {
  uint8_t state[16];

  uint tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = 0; i < 16; i++) {
    state[i] = input[tid * 16 + i];
  }

  AddRoundKeyGpu(state, expanded_key, 0);

  for (int i = 1; i < 14; i++) {
    SubBytesGpu(state, s);
    ShiftRowsGpu(state);
    MixColumnsGpu(state);
    AddRoundKeyGpu(state, expanded_key, i);
  }

  SubBytesGpu(state, s);
  ShiftRowsGpu(state);
  AddRoundKeyGpu(state, expanded_key, 14);

  for (int i = 0; i < 16; i++) {
    input[tid * 16 + i] = state[i];
  }
}

void AesCudaBenchmark::Run() {
  int num_blocks = text_length_ / 16;

  dim3 grid_size(static_cast<size_t>(num_blocks / 64.00));
  dim3 block_size(64);

  aes_cuda<<<grid_size, block_size>>>(dev_ciphertext_, dev_key_, dev_s_);

  cudaDeviceSynchronize();

  memcpy(ciphertext_, dev_ciphertext_, text_length_);
}

void AesCudaBenchmark::Cleanup() {
  AesBenchmark::Cleanup();
  cudaFree(dev_ciphertext_);
  cudaFree(dev_key_);
}
