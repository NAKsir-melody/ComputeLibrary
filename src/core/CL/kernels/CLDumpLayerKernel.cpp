/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "arm_compute/core/CL/kernels/CLDumpLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/CL/OpenCL.h"

#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLDumpLayerKernel::CLDumpLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLDumpLayerKernel::configure(ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    auto_init_if_empty(*output->info(), *input->info()->clone());

    _input          = input;
    _output         = output;
}

#include <time.h>
void CLDumpLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_MISMATCHING_WINDOWS(ICLKernel::window(), window);
    struct timespec begin, end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);

    _input->map(queue,true);
    _output->map(queue,true);
    ITensorInfo *src_info = _input->info();
    ITensorInfo *dst_info = _output->info();

    ARM_COMPUTE_ERROR_ON(src_info->num_dimensions() > dst_info->num_dimensions());
    ARM_COMPUTE_ERROR_ON(src_info->num_channels() != dst_info->num_channels());
    ARM_COMPUTE_ERROR_ON(src_info->element_size() != dst_info->element_size());

    for(size_t d = 0; d < src_info->num_dimensions(); d++)
    {
        ARM_COMPUTE_ERROR_ON(src_info->dimension(d) > dst_info->dimension(d));
    }

    // Copy information about valid region
    dst_info->set_valid_region(src_info->valid_region());

    Window win_src;
    win_src.use_tensor_dimensions(src_info->tensor_shape(), Window::DimY);
    Window win_dst;
    win_dst.use_tensor_dimensions(dst_info->tensor_shape(), Window::DimY);

    Iterator src_it(_input, win_src);
    Iterator dst_it(_output, win_dst);

    const size_t line_size = src_info->num_channels() * src_info->element_size() * src_info->dimension(0);
    std::cout << line_size << std::endl;


    execute_window_loop(win_src, [&](const Coordinates & id)
    {
        memcpy(dst_it.ptr(), src_it.ptr(), line_size);
    },
    src_it, dst_it);

    _input->unmap(queue);
    _output->unmap(queue);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    uint64_t time = 1e9 * (end.tv_sec - begin.tv_sec) + (end.tv_nsec -begin.tv_nsec);
    std::cout<< "Map & Copy" << time << "ns" << std::endl;
}
