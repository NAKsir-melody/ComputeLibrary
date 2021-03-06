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
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistry.h"

#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"

namespace arm_compute
{
namespace graph
{
namespace detail
{
void default_initialize_backends()
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        backend.second->initialize_backend();
    }
}

void validate_all_nodes(Graph &g)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    auto &nodes = g.nodes();

    // Create tasks
    // 백엔드에게 모든 노드를 하나하나 검증요청한다
    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target assigned_target = node->assigned_target();
            auto   backend         = backends::BackendRegistry::get().find_backend(assigned_target);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
            Status status = backend->validate_node(*node);
            ARM_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
        }
    }
}
// 타겟 백엔드에 텐서를 생성한다
void configure_all_tensors(Graph &g)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor)
        {
            Target target  = tensor->desc().target;
            auto   backend = backends::BackendRegistry::get().find_backend(target);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
            auto handle = backend->create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }
}

void allocate_all_input_tensors(INode &node)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        Tensor *tensor = node.input(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_all_output_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *tensor = node.output(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_const_tensors(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
            }
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr && tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
        {
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    auto &nodes = g.nodes();

    // Create tasks
    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target assigned_target = node->assigned_target();
            auto   backend         = backends::BackendRegistry::get().find_backend(assigned_target);
            ARM_COMPUTE_ERROR_ON_MSG(!backend, "Requested backend doesn't exist!");
	    // CL커널을 준비한다
            auto func = backend->configure_node(*node, ctx);
            if(func != nullptr)
            {
                ExecutionTask task;
                task.task = std::move(func);
                task.node = node.get();
                workload.tasks.push_back(std::move(task));
            }
        }
    }

    // Add inputs and outputs
    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            continue;
        }
    }

    return workload;
}

void release_unused_tensors(Graph &g)
{
    for(auto &tensor : g.tensors())
    {
        if(tensor != nullptr && tensor->handle() != nullptr)
        {
            tensor->handle()->release_if_unused();
        }
    }
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const)
        {
    		ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
            call_tensor_accessor(node->output(0));
        }
    }
}

void call_all_input_node_accessors(ExecutionWorkload &workload)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    for(auto &input : workload.inputs)
    {
        if(input != nullptr)
        {
            input->call_accessor();
        }
    }
}

void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    for(auto &task : workload.tasks)
    {
        task.prepare();
        release_unused_tensors(*workload.graph);
    }
}

void call_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);

    // Acquire memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }

    // Execute tasks
    for(auto &task : workload.tasks)
    {
	// enqueue(queue, task, slice);
	// operator overloading ()
	    std::cout<<task.node->name()<<std::endl;
        task();
    }

    // Release memory for the transition buffers
    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

void call_all_output_node_accessors(ExecutionWorkload &workload)
{
    ARM_COMPUTE_LOG_GRAPH_VERBOSE(__func__<<  std::endl);
    for(auto &output : workload.outputs)
    {
        if(output != nullptr)
        {
            output->call_accessor();
        }
    }
}
} // namespace detail
} // namespace graph
} // namespace arm_compute
