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

#include "CL/cl.h"
#include "stdio.h"
#include "hist_cl12_benchmark.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

#define SUCCESS 0
#define FAILURE 1

const int CACHE_LINE_SIZE = 64;
// OpenCL datastructures
cl_context       context_;
cl_device_id     *devices;
cl_device_id     device_;
cl_command_queue cmd_queue_;
cl_program       program_;
cl_kernel        hist_kernel_;

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

}

void HistCl12Benchmark::Initialize() {
  setupOpenCL();
  HistBenchmark::Initialize();

  InitializeKernels();
}

void HistCl12Benchmark::InitializeKernels() {
  cl_int err;

  err = clBuildProgram(program_, 1, devices, NULL, NULL, NULL);

  hist_kernel_ = clCreateKernel(program_, "HIST", &err);
}

void HistCl12Benchmark::Run() {
  cl_int err;

  memset(histogram_, 0, num_color_ * sizeof(uint32_t));

  err = clSetKernelArg(hist_kernel_, 0, sizeof(int *), &pixels_);
  err = clSetKernelArg(hist_kernel_, 1, sizeof(int *), &histogram_);
  err = clSetKernelArg(hist_kernel_, 2, sizeof(uint32_t), &num_color_);
  err = clSetKernelArg(hist_kernel_, 3, sizeof(uint32_t), &num_pixel_);

  size_t global_dimensions[] = {1024};
  size_t local_dimensions[] = {64};
  cl_event event;
  err = clEnqueueNDRangeKernel(cmd_queue_, hist_kernel_, CL_TRUE, NULL,
                               global_dimensions, local_dimensions, 0, 0, &event);
  err = clWaitForEvents(1, &event);
}

void HistCl12Benchmark::Cleanup() {
  HistBenchmark::Cleanup();

  cl_int ret;
  ret = clReleaseKernel(hist_kernel_);
  ret = clReleaseProgram(program_);
}
