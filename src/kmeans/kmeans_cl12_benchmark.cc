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

#include "kmeans_cl12_benchmark.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <malloc.h>

#include <cstdlib>
#include <memory>

#define SUCCESS 0
#define FAILURE 1

const int CACHE_LINE_SIZE = 64;
// OpenCL datastructures
cl_context       context_;
cl_device_id     *devices;
cl_device_id     device_;
cl_command_queue cmd_queue_;
cl_program       program_;
cl_kernel        kmeans_kernel_compute_;
cl_kernel        kmeans_kernel_swap_;

int *locPtr;

int setupOpenCL() {
  cl_int status = 0;
  size_t deviceListSize;

  // 1. Get platform
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS) {
    printf("Error: Getting Platforms. (clGetPlatformsIDs)\n");
    return FAILURE;
  }

  if (numPlatforms > 0) {
    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
      printf("Error: Getting Platform Ids. (clGetPlatformsIDs)\n");
      return FAILURE;
    }
    for (int i = 0; i < numPlatforms; ++i) {
      char pbuff[100];
      status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                                 sizeof(pbuff), pbuff, NULL);
      if (status != CL_SUCCESS) {
        printf("Error: Getting Platform Info.(clGetPlatformInfo)\n");
        return FAILURE;
      }
      printf("\n%s\n", pbuff);
      platform = platforms[i];
      if (!strcmp(pbuff, "Advanced Micro Devices, Inc.")) {
        break;
      }
    }
    delete platforms;
 }

  if (NULL == platform) {
    printf("NULL platform found so Exiting Application.\n");
    return FAILURE;
  }

  // 2. create context from platform
  cl_context_properties cps[3] =
          {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
  context_ = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL,
                                    &status);
  if (status != CL_SUCCESS) {
    printf("Error: Creating Context. (clCreateContextFromType)\n");
    return FAILURE;
  }

  // 3. Get device info
  // 3a. Get # of devices
  status = clGetContextInfo(context_, CL_CONTEXT_DEVICES, 0, NULL,
                            &deviceListSize);
  if (status != CL_SUCCESS) {
    printf("Error: Getting Context Info (1st clGetContextInfo)\n");
    return FAILURE;
  }

  // 3b. Get the device list data
  devices = (cl_device_id *)malloc(deviceListSize);
  if (devices == 0) {
    printf("Error: No devices found.\n");
    return FAILURE;
  }
  status = clGetContextInfo(context_, CL_CONTEXT_DEVICES, deviceListSize,
                            devices, NULL);
  if (status != CL_SUCCESS) {
    printf("Error: Getting Context Info (2nd clGetContextInfo)\n");
    return FAILURE;
  }

  device_ = devices[0];
  // 4. Create command queue for device
  cmd_queue_ = clCreateCommandQueue(context_, devices[0], 0, &status);
  if (status != CL_SUCCESS) {
    printf("Creating Command Queue. (clCreateCommandQueue)\n");
    return FAILURE;
  }

  const char *source = "dmmy text";

  size_t sourceSize[] = {strlen(source)};

  // 5b. Register the kernel with the runtime
  program_ = clCreateProgramWithSource(context_, 1, &source, sourceSize,
                                       &status);
  if (status != CL_SUCCESS) {
    printf("Error: Loading kernel (clCreateProgramWithSource)\n");
    return FAILURE;
  }

  locPtr = (int *)memalign(CACHE_LINE_SIZE, sizeof(int));
  *locPtr = 0;
}

void KmeansCl12Benchmark::Initialize() {
  KmeansBenchmark::Initialize();

  setupOpenCL();
  InitializeKernels();
  InitializeBuffers();
  InitializeData();
}

void KmeansCl12Benchmark::InitializeKernels() {
  cl_int err;

  err = clBuildProgram(program_, 1, devices, NULL, NULL, NULL);

  kmeans_kernel_compute_ = clCreateKernel(program_, "kmeans_kernel_compute", &err);

  kmeans_kernel_swap_ = clCreateKernel(program_, "kmeans_kernel_swap", &err);
}

void KmeansCl12Benchmark::InitializeBuffers() {
  // Create host buffers
  host_features_swap_ = new float[num_points_ * num_features_];
  membership_ = reinterpret_cast<int *>(new int[num_points_]);
}

void KmeansCl12Benchmark::Clustering() {
  min_rmse_ = FLT_MAX;

  printf("Reached %d\n",__LINE__);
  // Sweep k from min to max_clusters_ to find the best number of clusters
  for (num_clusters_ = min_num_clusters_; num_clusters_ <= max_num_clusters_;
       num_clusters_++) {
    // Sanity check: cannot have more clusters than points
    if (num_clusters_ > num_points_) break;

    TransposeFeatures();
    KmeansClustering(num_clusters_);
    float rmse = CalculateRMSE();
    if (rmse < min_rmse_) {
      min_rmse_ = rmse;
      best_num_clusters_ = num_clusters_;
    }
  }
}

void KmeansCl12Benchmark::TransposeFeatures() {
  cl_int err;
  cl_event event;

  clSetKernelArg(kmeans_kernel_swap_, 0, sizeof(float *),
                 reinterpret_cast<void *>(&host_features_));
  clSetKernelArg(kmeans_kernel_swap_, 1, sizeof(float *),
                 reinterpret_cast<void *>(&host_features_swap_));
  clSetKernelArg(kmeans_kernel_swap_, 2, sizeof(int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_swap_, 3, sizeof(int),
                 reinterpret_cast<void *>(&num_features_));

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_swap_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, &event);
  err = clWaitForEvents(1, &event);

  // std::unique_ptr<float[]> trans_result(
  //     new float[num_points_ * num_features_]());
  //
  // err = clEnqueueReadBuffer(cmd_queue_, device_features_swap_, CL_TRUE, 0,
  //                           sizeof(float) * num_points_ * num_features_,
  //                           trans_result.get(), 0, 0, NULL);
}

void KmeansCl12Benchmark::KmeansClustering(unsigned num_clusters) {
  int num_iteration = 0;

  // Sanity check: avoid a cluster without points
  if (num_clusters > num_points_) {
    std::cerr << "# of clusters < # of points" << std::endl;
    exit(-1);
  }

  InitializeClusters(num_clusters);
  InitializeMembership();

  // Iterate until convergence
  do {
    std::cout << "Start" << std::endl;
    UpdateMembership(num_clusters);
    std::cout << "Done updating membership" << std::endl;
    UpdateClusterCentroids(num_clusters);
    num_iteration++;
  } while ((delta_ > 0) && (num_iteration < 500));

  std::cout << "# of iterations: " << num_iteration << std::endl;
}

void KmeansCl12Benchmark::UpdateMembership(unsigned num_clusters) {
  std::unique_ptr<int[]> new_membership(new int[num_points_]());

  size_t global_work_size = (size_t)num_points_;
  size_t local_work_size = kBlockSize;

  if (global_work_size % local_work_size != 0)
    global_work_size =
        (global_work_size / local_work_size + 1) * local_work_size;

  cl_int err;

  int size = 0;
  int offset = 0;

  void * new_mem = new_membership.get();

  clSetKernelArg(kmeans_kernel_compute_, 0, sizeof(float *),
                 reinterpret_cast<void *>(&host_features_swap_));
  clSetKernelArg(kmeans_kernel_compute_, 1, sizeof(float *),
                 reinterpret_cast<void *>(&clusters_));
  clSetKernelArg(kmeans_kernel_compute_, 2, sizeof(int *),
                 reinterpret_cast<void *>(&new_mem));
  clSetKernelArg(kmeans_kernel_compute_, 3, sizeof(int),
                 reinterpret_cast<void *>(&num_points_));
  clSetKernelArg(kmeans_kernel_compute_, 4, sizeof(int),
                 reinterpret_cast<void *>(&num_clusters));
  clSetKernelArg(kmeans_kernel_compute_, 5, sizeof(int),
                 reinterpret_cast<void *>(&num_features_));
  clSetKernelArg(kmeans_kernel_compute_, 6, sizeof(int),
                 reinterpret_cast<void *>(&offset));
  clSetKernelArg(kmeans_kernel_compute_, 7, sizeof(int),
                 reinterpret_cast<void *>(&size));

  cl_event event;
  err = clEnqueueNDRangeKernel(cmd_queue_, kmeans_kernel_compute_, 1, NULL,
                               &global_work_size, &local_work_size, 0, 0, &event);

  err = clWaitForEvents(1, &event);
  //clFinish(cmd_queue_);

  delta_ = 0.0f;
  for (unsigned i = 0; i < num_points_; i++) {
    if (new_membership[i] != membership_[i]) {
      delta_++;
      membership_[i] = new_membership[i];
    }
  }
}

void KmeansCl12Benchmark::InitializeData() {}

void KmeansCl12Benchmark::Run() { Clustering(); }

void KmeansCl12Benchmark::Cleanup() {
  KmeansBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(kmeans_kernel_swap_);
  ret = clReleaseKernel(kmeans_kernel_compute_);
  ret = clReleaseProgram(program_);
}
