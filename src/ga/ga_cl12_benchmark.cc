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

#include "ga_cl12_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <openclsetup.h>

void GaCl12Benchmark::Initialize() {
  setupOpenCL();
  GaBenchmark::Initialize();

  InitializeKernels();
  InitializeBuffers();
}

void GaCl12Benchmark::InitializeKernels() {
  cl_int err;

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);

  ga_kernel_ = clCreateKernel(program_, "ga_cl12", &err);
}

void GaCl12Benchmark::InitializeBuffers() {
  GaBenchmark::Initialize();
  coarse_match_result_ = new char[target_sequence_.size()];
}

void GaCl12Benchmark::Run() {
  if (collaborative_) {
    CollaborativeRun();
  } else {
    NonCollaborativeRun();
  }
}

void GaCl12Benchmark::CollaborativeRun() {
  matches_.clear();

  cl_int err;
  uint32_t max_searchable_length =
      target_sequence_.size() - coarse_match_length_;
  std::vector<std::thread> threads;
  uint32_t current_position = 0;

  while (current_position < max_searchable_length) {
    char batch_result[kBatchSize] = {0};

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    size_t localThreads[1] = {64};
    size_t globalThreads[1] = {length};

    // std::cout << "localThreads: " << localThreads[1] << std::endl;
    // std::cout << "globalThreads: " << globalThreads[1] << std::endl;

    // Set the arguments of the kernel
    char *target_ = target_sequence_.data();
    err = clSetKernelArg(ga_kernel_, 0, sizeof(char *),
                         reinterpret_cast<void *>(&target_));

    char *query_ = query_sequence_.data();
    err = clSetKernelArg(ga_kernel_, 1, sizeof(char *),
                         reinterpret_cast<void *>(&query_));

    err = clSetKernelArg(ga_kernel_, 2, sizeof(char *),
                         reinterpret_cast<void *>(&batch_result));

    err = clSetKernelArg(ga_kernel_, 3, sizeof(uint),
                         reinterpret_cast<void *>(&length));

    int qsize = query_sequence_.size();
    err = clSetKernelArg(ga_kernel_, 4, sizeof(uint),
                         reinterpret_cast<void *>(&qsize));

    err = clSetKernelArg(ga_kernel_, 5, sizeof(int),
                         reinterpret_cast<void *>(&coarse_match_length_));

    err = clSetKernelArg(ga_kernel_, 6, sizeof(int),
                         reinterpret_cast<void *>(&coarse_match_threshold_));

    err = clSetKernelArg(ga_kernel_, 7, sizeof(int),
                         reinterpret_cast<void *>(&current_position));

    // Execute the OpenCL kernel on the list
    err = clEnqueueNDRangeKernel(cmd_queue_, ga_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);
    clFinish(cmd_queue_);

    for (uint32_t i = 0; i < length; i++) {
      if (batch_result[i] != 0) {
        uint32_t end = i + current_position + query_sequence_.size();
        if (end > target_sequence_.size()) end = target_sequence_.size();
        threads.push_back(std::thread(&GaCl12Benchmark::FineMatch, this,
                                      i + current_position, end, &matches_));
      }
    }
    current_position = end_position;
  }
  for (auto &thread : threads) {
    thread.join();
  }
}

void GaCl12Benchmark::NonCollaborativeRun() {
  matches_.clear();
  cl_int err;

  uint32_t max_searchable_length =
      target_sequence_.size() - coarse_match_length_;
#if 0
  std::vector<std::thread> threads;
#endif
  uint32_t current_position = 0;

  printf("@@@@@@@@@@@@@@@@@GPU Start@@@@@@@@@@@@@@@@@@\n");
  while (current_position < max_searchable_length) {
    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    // Set the arguments of the kernel
    char *target_ = target_sequence_.data();
    err = clSetKernelArg(ga_kernel_, 0, sizeof(char *),
                         reinterpret_cast<void *>(&target_));

    char *query_ = query_sequence_.data();
    err = clSetKernelArg(ga_kernel_, 1, sizeof(char *),
                         reinterpret_cast<void *>(&query_));

    char *bresult = coarse_match_result_ + current_position;
    err = clSetKernelArg(ga_kernel_, 2, sizeof(char *),
                         reinterpret_cast<void *>(&bresult));

    err = clSetKernelArg(ga_kernel_, 3, sizeof(uint),
                         reinterpret_cast<void *>(&length));

    int qsize = query_sequence_.size();
    err = clSetKernelArg(ga_kernel_, 4, sizeof(int),
                         reinterpret_cast<void *>(&qsize));

    err = clSetKernelArg(ga_kernel_, 5, sizeof(int),
                         reinterpret_cast<void *>(&coarse_match_length_));

    err = clSetKernelArg(ga_kernel_, 6, sizeof(int),
                         reinterpret_cast<void *>(&coarse_match_threshold_));

    err = clSetKernelArg(ga_kernel_, 7, sizeof(int),
                         reinterpret_cast<void *>(&current_position));

    size_t localThreads[1] = {64};
    size_t globalThreads[1] = {length};


    // Execute the OpenCL kernel on the list
    err = clEnqueueNDRangeKernel(cmd_queue_, ga_kernel_, CL_TRUE, NULL,
                                 globalThreads, localThreads, 0, NULL, NULL);

    clFinish(cmd_queue_);

    current_position = end_position;
  }
  printf("@@@@@@@@@@@@@@@@@GPU Over@@@@@@@@@@@@@@@@@@\n");

  for (uint32_t i = 0; i < target_sequence_.size(); i++) {
    if (coarse_match_result_[i] != 0) {
      uint32_t end = i + query_sequence_.size();
      if (end > target_sequence_.size()) end = target_sequence_.size();
#if 0
      threads.push_back(
          std::thread(&GaCl12Benchmark::FineMatch, this, i, end, &matches_));
#else
          FineMatch( i, end, &matches_);
#endif
    }
  }
  
  printf("@@@@@@@@@@@@@@@@@CPU End@@@@@@@@@@@@@@@@@@\n");
#if 0
  for (auto &thread : threads) {
    thread.join();
  }
#endif
}

void GaCl12Benchmark::Cleanup() {
  free(coarse_match_result_);
  GaBenchmark::Cleanup();
}
