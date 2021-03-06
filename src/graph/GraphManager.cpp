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
#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
namespace graph
{
GraphManager::GraphManager()
    : _workloads()
{
    detail::default_initialize_backends();
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    // Setup graph context if not done manually
    setup_default_graph_context(ctx);

    // Check if graph has been registered
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

    // Force target to all graph construct
    // node & tensor
    Target forced_target = is_target_supported(target) ? target : get_default_target();
    force_target_to_graph(graph, forced_target);

    // Configure all tensors
    // 타겟 백엔드에 텐서를 생성한다
    detail::configure_all_tensors(graph);

    // Apply all mutating passes
    // 모든 변형 경로를 실행한다
    pm.run_all(graph);

    // Validate all nodes
    // 백엔드에게 모든 노드를 하나하나 검증요청한다
    detail::validate_all_nodes(graph);

    // Configure all nodes
    // CL커널이 동작할수 있도록 모든 노드를 준비시킨다
    auto workload = detail::configure_all_nodes(graph, ctx);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    // Allocate const tensors and call accessors
    // clCreateBuffer
    detail::allocate_const_tensors(graph);
    
    // enqueueMapBuffer
    detail::call_all_const_node_accessors(graph);

    if(forced_target == Target::CL)
    {
        // Prepare graph
	// 상황에따라 메모리 레이아웃을 변경하거나, 준비커널을 실행해야 하는 경우가 있음 
	// gemm convolution의 경우 transpose커널을 한번 실행함
        detail::prepare_all_tasks(workload);
    }

    // Setup tensor memory (Allocate all tensors or setup transition manager)
    if(ctx.config().use_transition_memory_manager)
    {
        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }

    // Finalize Graph context
    ctx.finalize();

    // Register graph
    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id().get() << std::endl);

    if(forced_target != Target::CL)
    {
        // Make first run
        execute_graph(graph);

        // Release all unused const tensors
        detail::release_unused_tensors(graph);
    }
}

void GraphManager::execute_graph(Graph &graph)
{
    // Check if graph is finalized
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("execute graph with ID : " << graph.id().get() << std::endl);
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    // Call input accessors
    detail::call_all_input_node_accessors(it->second);

    // Run graph
	auto queue = arm_compute::CLScheduler::get().queue();
	cl_command_queue_properties props = queue.getInfo<CL_QUEUE_PROPERTIES>();
	arm_compute::CLScheduler::get().set_queue(cl::CommandQueue(arm_compute::CLScheduler::get().context(), props | CL_QUEUE_PROFILING_ENABLE));
	cl::Event start, stop;
	queue.enqueueMarker(&start);
    detail::call_all_tasks(it->second);
	queue.enqueueMarker(&stop);
	stop.wait();

	cl_ulong time_start, time_end;
	double total_time;
	start.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_start);
	stop.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_end);
	total_time = time_end - time_start;
	std::cout << "cltime" << total_time <<"ns"  <<std::endl;

    // Call output accessors
    detail::call_all_output_node_accessors(it->second);
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} // namespace graph
} // namespace arm_compute
