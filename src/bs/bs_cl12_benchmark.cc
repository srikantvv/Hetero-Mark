/*
 * Hetero-mark
 *
 * Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:
 *   Northeastern University Computer Architecture Research (NUCAR) Group
 *   Northeastern University
 *   http://www.ece.neu.edu/groups/nucar/
 *
 * Author: Yifan Sun (yifansun@coe.neu.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *   Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimers.
 *
 *   Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   Neither the names of NUCAR, Northeastern University, nor the names of
 *   its contributors may be used to endorse or promote products derived
 *   from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include "stdint.h"
#include "bs_cl12_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <openclsetup.h>

void BsCl12Benchmark::Initialize() {
  setupOpenCL();
  cl_int err;
  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);
  bs_kernel_ = clCreateKernel(program_, "blackscholes", &err);

  BsBenchmark::Initialize();

  for (unsigned int i = 0; i < num_tiles_ * tile_size_; i++) {
    call_price_[i] = 0;
    put_price_[i] = 0;
  }

}

void BsCl12Benchmark::Run() {
  // The main while loop
  uint32_t done_tiles_ = 0;
  uint32_t last_tile_ = num_tiles_;

  cl_event event = NULL;
  // while the done tiles are less than num_tiles, continue
  while (done_tiles_ < last_tile_) {
    // First check to make sure that we are launching the first set
    if (IsGpuCompleted(event)) {
      // No longer the first lunch after this point so
      // turn it off
      // printf("Completion set to 1. GPU running \n");

      // Set the size of the section based on the number of tiles
      // and the number of compute units
      uint32_t section_tiles = (gpu_chunk_ < last_tile_ - done_tiles_)
                                   ? gpu_chunk_
                                   : last_tile_ - done_tiles_;

      unsigned int offset = done_tiles_ * tile_size_;
      //               printf("Section tile is %d \n", section_tiles);

      // GPU is running the following tiles
      // fprintf(stderr, "GPU tiles: %d to %d\n", done_tiles_,
      //       done_tiles_ + section_tiles);
      done_tiles_ += section_tiles;

      size_t global_work_size[] = {(section_tiles * tile_size_) / 64};
      size_t local_work_size[] = {64};

      printf("global:%d local:%d\n", global_work_size[0], local_work_size[0]);
      cl_int err;

      err = clSetKernelArg(bs_kernel_, 0, sizeof(float *), &rand_array_ + offset);
      err = clSetKernelArg(bs_kernel_, 1, sizeof(float *), &call_price_ + offset);
      err = clSetKernelArg(bs_kernel_, 2, sizeof(float *), &put_price_ + offset);

      err = clEnqueueNDRangeKernel(cmd_queue_, bs_kernel_, 1, NULL,
                                 global_work_size, local_work_size, 0, NULL,
                                 &event);
    } else {
      if (active_cpu_) {
        last_tile_--;
        // fprintf(stderr, "CPU tile: %d \n", last_tile_);
        BlackScholesCPU(rand_array_, call_price_, put_price_,
                        last_tile_ * tile_size_, tile_size_);
      }
    }
  }

  clFinish(cmd_queue_);
}

bool BsCl12Benchmark::IsGpuCompleted(cl_event check_event) {
  cl_int err;
  cl_int ret;
  err = clGetEventInfo(check_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                          sizeof(cl_int), &ret, NULL);
  if (ret == CL_COMPLETE) return true;
  return false;
}

void BsCl12Benchmark::Cleanup() {
  cl_int ret;
  ret = clReleaseKernel(bs_kernel_);
  ret = clReleaseProgram(program_);
  BsBenchmark::Cleanup();
}
