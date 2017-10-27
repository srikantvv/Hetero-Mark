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

#include <stdint.h>
#include "ep_cl12_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include "openclsetup.h"

void EpCl12Benchmark::Initialize() {
  setupOpenCL();
  EpBenchmark::Initialize();

  InitializeKernels();
}

void EpCl12Benchmark::InitializeKernels() {
  cl_int err;

  err = clBuildProgram(program_, 0, NULL, NULL, NULL, NULL);

  Evaluate_Kernel_ = clCreateKernel(program_, "Evaluate_Kernel", &err);

  Mutate_Kernel_ = clCreateKernel(program_, "Mutate_Kernel", &err);
}

void EpCl12Benchmark::Run() {
  if (pipelined_) {
    PipelinedRun();
  } else {
    NormalRun();
  }
}

void EpCl12Benchmark::PipelinedRun() {
  seed_ = kSeedInitValue;
  ReproduceInIsland(&islands_1_);
  for (uint32_t i = 0; i < max_generation_; i++) {
    printf("Stage 1 Start\n");
    std::thread t1(&EpCl12Benchmark::ReproduceInIsland, this, &islands_2_);
    std::thread t2(&EpCl12Benchmark::EvaluateGpu, this, &islands_1_);
    t1.join();
    t2.join();
    printf("Stage 1 End\n");

    printf("Stage 2 Start\n");
    std::thread t3(&EpCl12Benchmark::EvaluateGpu, this, &islands_2_);
    std::thread t4(&EpCl12Benchmark::SelectInIsland, this, &islands_1_);
    t4.join();
    result_island_1_ = islands_1_[0].fitness;
    std::thread t5(&EpCl12Benchmark::CrossoverInIsland, this, &islands_1_);
    t5.join();
    t3.join();
    printf("Stage 2 End\n");

    printf("Stage 3 Start\n");
    std::thread t6(&EpCl12Benchmark::SelectInIsland, this, &islands_2_);
    std::thread t7(&EpCl12Benchmark::MutateGpu, this, &islands_1_);
    t6.join();
    result_island_2_ = islands_2_[0].fitness;
    std::thread t8(&EpCl12Benchmark::CrossoverInIsland, this, &islands_2_);
    t7.join();
    t8.join();
    printf("Stage 3 End\n");

    printf("Stage 4 Start\n");
    std::thread t9(&EpCl12Benchmark::MutateGpu, this, &islands_2_);
    std::thread t10(&EpCl12Benchmark::ReproduceInIsland, this, &islands_1_);
    t9.join();
    t10.join();
    printf("Stage 4 End\n");

  }
}

void EpCl12Benchmark::NormalRun() {
  seed_ = kSeedInitValue;
  for (uint32_t i = 0; i < max_generation_; i++) {
    Reproduce();
    EvaluateGpu(&islands_1_);
    EvaluateGpu(&islands_2_);
    Select();

    result_island_1_ = islands_1_[0].fitness;
    result_island_2_ = islands_2_[0].fitness;

    Crossover();
    MutateGpu(&islands_1_);
    MutateGpu(&islands_2_);
  }
}

void EpCl12Benchmark::EvaluateGpu(std::vector<Creature> *island) {
  cl_int ret;

  size_t localThreads[1] = {64};
  size_t globalThreads[1] = {population_ / 2};

  Creature *creature_data = island->data();
  double *ffunc = &fitness_function_[0];
  // Set kernel arguments
  ret = clSetKernelArg(Evaluate_Kernel_, 0, sizeof(Creature *),
		       reinterpret_cast<void *>(&creature_data));

  ret = clSetKernelArg(Evaluate_Kernel_, 1, sizeof(double *),
		       reinterpret_cast<void *>(&ffunc));

  uint32_t half_population_ = population_ / 2;
  
  ret = clSetKernelArg(Evaluate_Kernel_, 2, sizeof(int),
		       reinterpret_cast<void *>(&half_population_));

  int num_var = kNumVariables;
  ret = clSetKernelArg(Evaluate_Kernel_, 3, sizeof(int),
		       const_cast<void*>(reinterpret_cast<const void *>(&num_var)));

  // Launch kernel
  ret = clEnqueueNDRangeKernel(cmd_queue_, Evaluate_Kernel_, CL_TRUE, NULL,
			       globalThreads, localThreads, 0, NULL, NULL);
  clFinish(cmd_queue_);
}

void EpCl12Benchmark::MutateGpu(std::vector<Creature> *island) {
  cl_int ret;
  
  size_t localThreads[1] = {64};
  size_t globalThreads[1] = {population_ / 2};

  Creature *creature_data = island->data();
// Set kernel arguments
  ret = clSetKernelArg(Mutate_Kernel_, 0, sizeof(Creature *),
		       reinterpret_cast<void *>(&creature_data));

  uint32_t half_population_ = population_ / 2;
  
  ret = clSetKernelArg(Mutate_Kernel_, 1, sizeof(int),
		       reinterpret_cast<void *>(&half_population_));

  int num_var = kNumVariables;
  ret = clSetKernelArg(Mutate_Kernel_, 2, sizeof(int),
		       const_cast<void*>(reinterpret_cast<const void *>(&num_var)));
  
  // Launch kernel
  ret = clEnqueueNDRangeKernel(cmd_queue_, Mutate_Kernel_, CL_TRUE, NULL,
			       globalThreads, localThreads, 0, NULL, NULL);
  clFinish(cmd_queue_);
}

void EpCl12Benchmark::Cleanup() {
  cl_int ret;

  ret = clReleaseKernel(Evaluate_Kernel_);
  ret = clReleaseKernel(Mutate_Kernel_);
  ret = clReleaseProgram(program_);

  EpBenchmark::Cleanup();
}
