/* Copyright (c) 2015 Northeastern University
 * All rights reserved.
 *
 * Developed by:Northeastern University Computer Architecture Research (NUCAR)
 * Group, Northeastern University, http://www.ece.neu.edu/groups/nucar/
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

#include <CL/cl.h>
#include "pr_cl12_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <openclsetup.h>

cl_kernel        pr_kernel_;

void PrCl12Benchmark::Initialize() {
  setupOpenCL();
  PrBenchmark::Initialize();

  InitializeKernels();
  InitializeBuffers();
}

void PrCl12Benchmark::InitializeKernels() {
  cl_int err;

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);

  pr_kernel_ = clCreateKernel(program_, "PageRankUpdateGpu", &err);
}

void PrCl12Benchmark::InitializeBuffers() {
  cl_int err;

  dev_page_rank_ = new float[num_nodes_];
  dev_page_rank_temp_ = new float[num_nodes_];

}

void PrCl12Benchmark::CopyDataToDevice() {
  cl_int err;

  for (uint32_t i = 0; i < num_nodes_; i++) {
    dev_page_rank_[i] = 1.0 / num_nodes_;
  }

}

void PrCl12Benchmark::CopyDataBackFromDevice(float *buffer) {
  memcpy(page_rank_, buffer, num_nodes_ * sizeof(float));
}

void PrCl12Benchmark::Run() {
  CopyDataToDevice();

  cl_int err;

  err = clSetKernelArg(pr_kernel_, 0, sizeof(uint32_t), &num_nodes_);

  err = clSetKernelArg(pr_kernel_, 1, sizeof(int *), &row_offsets_);

  err = clSetKernelArg(pr_kernel_, 2, sizeof(int *), &column_numbers_);

  err = clSetKernelArg(pr_kernel_, 3, sizeof(float *), &values_);

  err = clSetKernelArg(pr_kernel_, 4, sizeof(float) * 64, NULL);

  uint32_t i;
  //cl_event *event_list = (cl_event*)malloc(max_iteration_*(sizeof(cl_event)));;
  for (i = 0; i < max_iteration_; i++) {
    if (i % 2 == 0) {
      err = clSetKernelArg(pr_kernel_, 5, sizeof(float *), &dev_page_rank_);

      err = clSetKernelArg(pr_kernel_, 6, sizeof(float *), &dev_page_rank_temp_);
    } else {
      err = clSetKernelArg(pr_kernel_, 5, sizeof(float *), &dev_page_rank_temp_);

      err = clSetKernelArg(pr_kernel_, 6, sizeof(float *), &dev_page_rank_);
    }

    size_t global_work_size[] = {num_nodes_ * 64};
    size_t local_work_size[] = {64};
    //err = clEnqueueNDRangeKernel(cmd_queue_, pr_kernel_, 1, NULL,
                                 //global_work_size, local_work_size, 0, NULL,
                                 //&event_list[i]);
    err = clEnqueueNDRangeKernel(cmd_queue_, pr_kernel_, 1, NULL,
                                 global_work_size, local_work_size, 0, NULL,
                                 NULL);
  }

  clFinish(cmd_queue_);
  //for (i = 0; i < max_iteration_; i++) {
      //err = clWaitForEvents(1, &event_list[i]);
  //}
  //free(event_list);

  if (!i % 2 == 0) {
    CopyDataBackFromDevice(dev_page_rank_);
  } else {
    CopyDataBackFromDevice(dev_page_rank_temp_);
  }
}

void PrCl12Benchmark::Cleanup() {
  PrBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(pr_kernel_);
  ret = clReleaseProgram(program_);

  free(dev_page_rank_);
  free(dev_page_rank_temp_);

}

