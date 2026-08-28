/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#include "hesiod/model/graph/metal_graph_execution.hpp"

#include <algorithm>
#include <cstdlib>
#include <format>
#include <stdexcept>
#include <typeinfo>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

#include "hesiod/logger.hpp"
#include "hesiod/model/nodes/base_node.hpp"

namespace hesiod
{

namespace
{

thread_local MetalGraphExecution *current_execution = nullptr;
thread_local MetalGraphMetrics    last_metrics_value;

bool is_false_value(const char *value)
{
  if (!value) return false;
  const std::string setting(value);
  return setting == "0" || setting == "false" || setting == "FALSE" ||
         setting == "off" || setting == "OFF";
}

struct ProcessMemoryMetrics
{
  std::uint64_t rss_bytes = 0;
  std::uint64_t peak_rss_bytes = 0;
};

ProcessMemoryMetrics process_memory_metrics()
{
#if defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(),
                MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info),
                &count) == KERN_SUCCESS)
    return {static_cast<std::uint64_t>(info.resident_size),
            static_cast<std::uint64_t>(info.resident_size_max)};
#endif
  return {};
}

} // namespace

MetalGraphExecution::MetalGraphExecution(std::string graph_id)
    : graph_id_(std::move(graph_id))
{
  if (!environment_enabled())
    return;

  try
  {
    this->enabled_ = hmap::gpu::metal::is_available();
    if (this->enabled_)
      this->session_ = std::make_unique<hmap::gpu::metal::DeviceSession>(
          hmap::gpu::metal::StorageMode::shared);
  }
  catch (const std::exception &e)
  {
    this->enabled_ = false;
    Logger::log()->warn("Metal graph execution disabled for graph '{}': {}",
                        this->graph_id_,
                        e.what());
  }
}

MetalGraphExecution::~MetalGraphExecution()
{
  if (this->session_)
  {
    try
    {
      this->session_->finish();
    }
    catch (...)
    {
      // Destruction is best effort. Node failures are reported by BaseNode.
    }
  }
}

MetalGraphExecution *MetalGraphExecution::current() { return current_execution; }

const MetalGraphMetrics &MetalGraphExecution::last_metrics()
{
  return last_metrics_value;
}

bool MetalGraphExecution::environment_enabled()
{
  return !is_false_value(std::getenv("HESIOD_METAL_RESIDENT"));
}

bool MetalGraphExecution::resident_candidate(const std::string &node_type)
{
  return node_type == "CoherentNoise" || node_type == "SpectralEqualizer" ||
         node_type == "Thermal" || node_type == "Blend";
}

void MetalGraphExecution::prepare_node(BaseNode &node)
{
  if (!this->enabled_)
    return;

  if (!resident_candidate(node.get_node_type()))
    this->prepare_host_node(node);
}

void MetalGraphExecution::prepare_host_node(BaseNode &node)
{
  if (!this->enabled_)
    return;

  for (int port = 0; port < node.get_nports(); ++port)
  {
    if (node.get_data_type(port) != typeid(hmap::VirtualArray).name())
      continue;

    auto *array = static_cast<hmap::VirtualArray *>(node.get_data_ref(port));
    if (array && this->contains(array) && this->device_modified_.contains(array))
      this->materialize(array);
  }

  this->record_host(node, "host boundary");
}

hmap::gpu::metal::DeviceArray MetalGraphExecution::device_for(
    const hmap::VirtualArray *array)
{
  if (!this->enabled_ || !this->session_)
    throw std::runtime_error("Metal graph execution is disabled");
  if (!array)
    throw std::invalid_argument("Cannot upload a null VirtualArray");

  auto it = this->device_arrays_.find(array);
  if (it != this->device_arrays_.end())
    return it->second;

  const hmap::ComputeMode single_array_mode = {
      .mode = hmap::ForEachMode::VA_SINGLE_ARRAY,
      .trim_storage = false};
  const hmap::Array host = array->to_array(single_array_mode);
  auto              device = this->session_->upload(host);
  this->host_uploads_++;
  this->host_upload_bytes_ += host.vector.size() * sizeof(float);
  this->device_arrays_[array] = device;
  this->host_required_[array] = false;
  return device;
}

void MetalGraphExecution::bind(const hmap::VirtualArray     *array,
                               hmap::gpu::metal::DeviceArray device,
                               bool                          host_required)
{
  if (!this->enabled_ || !array)
    return;

  this->device_arrays_[array] = std::move(device);
  this->host_required_[array] = host_required;
  this->device_modified_.insert(array);
}

bool MetalGraphExecution::contains(const hmap::VirtualArray *array) const
{
  return array && this->device_arrays_.contains(array);
}

void MetalGraphExecution::materialize(const hmap::VirtualArray *array)
{
  if (!this->enabled_ || !array)
    return;

  auto it = this->device_arrays_.find(array);
  if (it == this->device_arrays_.end())
    return;

  const hmap::Array host = this->session_->download(it->second);
  auto *mutable_array = const_cast<hmap::VirtualArray *>(array);
  const hmap::ComputeMode single_array_mode = {
      .mode = hmap::ForEachMode::VA_SINGLE_ARRAY,
      .trim_storage = false};
  mutable_array->from_array(host, single_array_mode);
  this->host_readbacks_++;
  this->host_readback_bytes_ += host.vector.size() * sizeof(float);
  this->device_arrays_.erase(it);
  this->host_required_.erase(array);
  this->device_modified_.erase(array);
}

hmap::gpu::metal::DeviceSession &MetalGraphExecution::session()
{
  if (!this->session_)
    throw std::runtime_error("Metal graph execution is disabled");
  return *this->session_;
}

void MetalGraphExecution::record_resident(const BaseNode       &node,
                                          const std::string &detail)
{
  this->node_executions_.push_back(
      {node.get_id(), node.get_node_type(), "resident Metal", detail});
}

void MetalGraphExecution::record_host(const BaseNode       &node,
                                      const std::string &detail)
{
  this->node_executions_.push_back(
      {node.get_id(), node.get_node_type(), "host fallback", detail});
}

void MetalGraphExecution::flush()
{
  if (this->flushed_)
    return;

  if (!this->enabled_ || !this->session_)
  {
    this->capture_metrics();
    this->flushed_ = true;
    return;
  }

  std::vector<const hmap::VirtualArray *> outputs;
  for (const auto &[array, required] : this->host_required_)
    if (required)
      outputs.push_back(array);

  for (const auto *array : outputs)
    this->materialize(array);

  this->session_->finish();
  this->capture_metrics();
  this->flushed_ = true;
}

void MetalGraphExecution::capture_metrics()
{
  last_metrics_value = {};
  last_metrics_value.graph_id = this->graph_id_;
  last_metrics_value.enabled = this->enabled_ && this->session_ != nullptr;
  if (last_metrics_value.enabled)
  {
    last_metrics_value.device = hmap::gpu::metal::device_name();
    last_metrics_value.recommended_max_working_set_bytes =
        hmap::gpu::metal::capabilities().recommended_max_working_set_size;
  }
  const auto process_memory = process_memory_metrics();
  last_metrics_value.process_rss_bytes = process_memory.rss_bytes;
  last_metrics_value.process_peak_rss_bytes = process_memory.peak_rss_bytes;
  last_metrics_value.host_uploads = this->host_uploads_;
  last_metrics_value.host_readbacks = this->host_readbacks_;
  last_metrics_value.host_upload_bytes = this->host_upload_bytes_;
  last_metrics_value.host_readback_bytes = this->host_readback_bytes_;
  last_metrics_value.node_executions = this->node_executions_;
  last_metrics_value.resident_nodes = std::count_if(
      this->node_executions_.begin(),
      this->node_executions_.end(),
      [](const auto &node) { return node.backend == "resident Metal"; });
  last_metrics_value.host_nodes = std::count_if(
      this->node_executions_.begin(),
      this->node_executions_.end(),
      [](const auto &node) { return node.backend == "host fallback"; });
  if (this->session_)
    last_metrics_value.metal_stats = this->session_->stats();
}

std::string MetalGraphExecution::diagnostics() const
{
  if (!this->enabled_ || !this->session_)
    return "backend=disabled";

  const auto stats = this->session_->stats();
  return std::format(
      "graph={} backend=Metal device={} resident_nodes={} host_nodes={} "
      "uploads={} upload_bytes={} readbacks={} readback_bytes={} "
      "command_buffers={} encoders={} synchronizations={} gpu_ms={:.3f}",
      this->graph_id_,
      hmap::gpu::metal::device_name(),
      std::count_if(this->node_executions_.begin(),
                    this->node_executions_.end(),
                    [](const auto &node) { return node.backend == "resident Metal"; }),
      std::count_if(this->node_executions_.begin(),
                    this->node_executions_.end(),
                    [](const auto &node) { return node.backend == "host fallback"; }),
      this->host_uploads_,
      this->host_upload_bytes_,
      this->host_readbacks_,
      this->host_readback_bytes_,
      stats.command_buffers,
      stats.encoders,
      stats.synchronization_count,
      stats.gpu_execution_ms);
}

MetalGraphExecutionScope::MetalGraphExecutionScope(MetalGraphExecution &execution)
    : previous_(current_execution)
{
  current_execution = &execution;
}

MetalGraphExecutionScope::~MetalGraphExecutionScope()
{
  current_execution = this->previous_;
}

} // namespace hesiod
