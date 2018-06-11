/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph/Workload.h"

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorHandle.h"

namespace arm_compute
{
namespace graph
{
void ExecutionTask::operator()()
{
// enqueue(queue, task, slice);
    TaskExecutor::get().execute_function(*this);
}

void execute_task(ExecutionTask &task)
{
// enqueue(queue, task, slice);
    if(task.task)
    {
        task.task->run();
    }
}

void ExecutionTask::prepare()
{
    if(task)
    {
        task->prepare();
    }
}

TaskExecutor::TaskExecutor()
    : execute_function(execute_task)
{
}

TaskExecutor &TaskExecutor::get()
{
    static TaskExecutor executor;
    return executor;
}
} // namespace graph
} // namespace arm_compute
