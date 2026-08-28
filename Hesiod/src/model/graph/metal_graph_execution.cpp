/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#include "hesiod/model/graph/metal_graph_execution.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <iterator>
#include <limits>
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

bool is_true_value(const char *value)
{
  if (!value) return false;
  std::string setting(value);
  std::transform(setting.begin(),
                 setting.end(),
                 setting.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return setting == "1" || setting == "true" || setting == "on";
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

MetalGraphCache::MetalGraphCache()
{
  this->enabled_ = is_true_value(std::getenv("HESIOD_METAL_PERSISTENT_CACHE"));
  if (!this->enabled_)
    return;

  const auto capabilities = hmap::gpu::metal::capabilities();
  this->stats_.budget_bytes = capabilities.recommended_max_working_set_size / 4;

  if (const char *megabytes = std::getenv("HESIOD_METAL_CACHE_MB"))
  {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(megabytes, &end, 10);
    if (end != megabytes && value > 0 &&
        value <= std::numeric_limits<std::size_t>::max() / (1024ull * 1024ull))
      this->stats_.budget_bytes = static_cast<std::size_t>(value) * 1024u * 1024u;
  }
}

hmap::gpu::metal::DeviceArray MetalGraphCache::acquire(
    hmap::gpu::metal::DeviceSession &session,
    const hmap::VirtualArray        *array)
{
  if (!this->enabled_ || !array)
    return {};

  auto it = this->entries_.find(array);
  if (it == this->entries_.end() ||
      it->second.shape != array->shape ||
      it->second.tile_shape != array->tile_shape ||
      it->second.halo != array->halo)
  {
    if (it != this->entries_.end())
    {
      this->stats_.bytes -= it->second.bytes;
      this->entries_.erase(it);
    }
    ++this->stats_.misses;
    return {};
  }

  try
  {
    auto adopted = session.adopt_completed(it->second.device);
    it->second.device = adopted;
    it->second.last_use = ++this->clock_;
    ++this->stats_.hits;
    return adopted;
  }
  catch (...)
  {
    this->stats_.bytes -= it->second.bytes;
    this->entries_.erase(it);
    ++this->stats_.misses;
    return {};
  }
}

void MetalGraphCache::evict_until_within_budget(std::size_t required_bytes)
{
  while (this->stats_.bytes + required_bytes > this->stats_.budget_bytes &&
         !this->entries_.empty())
  {
    auto oldest = this->entries_.begin();
    for (auto it = std::next(this->entries_.begin()); it != this->entries_.end(); ++it)
      if (it->second.last_use < oldest->second.last_use)
        oldest = it;

    this->stats_.bytes -= oldest->second.bytes;
    this->entries_.erase(oldest);
    ++this->stats_.evictions;
  }
}

void MetalGraphCache::store(const hmap::VirtualArray *array,
                            const hmap::gpu::metal::DeviceArray &device)
{
  if (!this->enabled_ || !array || device.empty())
    return;

  const std::size_t bytes = device.size() * sizeof(float);
  auto existing = this->entries_.find(array);
  if (existing != this->entries_.end())
  {
    this->stats_.bytes -= existing->second.bytes;
    this->entries_.erase(existing);
  }

  if (bytes > this->stats_.budget_bytes)
  {
    ++this->stats_.evictions;
    return;
  }

  this->evict_until_within_budget(bytes);
  this->entries_.emplace(array,
                         Entry{device,
                               array->shape,
                               array->tile_shape,
                               array->halo,
                               bytes,
                               ++this->clock_});
  this->stats_.bytes += bytes;
}

void MetalGraphCache::invalidate(const hmap::VirtualArray *array)
{
  if (!array) return;
  auto it = this->entries_.find(array);
  if (it == this->entries_.end()) return;
  this->stats_.bytes -= it->second.bytes;
  this->entries_.erase(it);
}

void MetalGraphCache::clear()
{
  this->entries_.clear();
  this->stats_.bytes = 0;
}

MetalGraphExecution::MetalGraphExecution(std::string graph_id,
                                         std::shared_ptr<MetalGraphCache> cache)
    : graph_id_(std::move(graph_id)), cache_(std::move(cache))
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
         node_type == "Thermal" || node_type == "Blend" ||
         node_type == "GaborWaveFbm" || node_type == "MorphologicalGradient";
}

void MetalGraphExecution::prepare_node(BaseNode &node)
{
  if (!this->enabled_)
    return;

  if (this->cache_ && this->cache_->enabled())
    for (int port = 0; port < node.get_nports(); ++port)
    {
      if (node.get_port_type(port) != gngui::PortType::OUT ||
          node.get_data_type(port) != typeid(hmap::VirtualArray).name())
        continue;
      this->cache_->invalidate(
          static_cast<hmap::VirtualArray *>(node.get_data_ref(port)));
    }

  if (!resident_candidate(node.get_node_type()))
    this->prepare_host_node(node);
}

void MetalGraphExecution::prepare_host_node(BaseNode       &node,
                                            const std::string &detail)
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

  this->record_host(node, detail);
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

  if (this->cache_ && this->cache_->enabled())
  {
    auto cached = this->cache_->acquire(*this->session_, array);
    if (!cached.empty())
    {
      this->device_arrays_[array] = cached;
      this->host_required_[array] = false;
      return cached;
    }
  }

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
  this->session_finished_ = true;
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
  for (int port = 0; port < node.get_nports(); ++port)
    if (node.get_port_type(port) == gngui::PortType::OUT &&
        node.get_data_type(port) == typeid(hmap::VirtualArray).name())
      if (auto *array = static_cast<hmap::VirtualArray *>(
              const_cast<BaseNode &>(node).get_data_ref(port)))
        this->resident_tiles_ += static_cast<std::size_t>(array->get_ntiles());
}

void MetalGraphExecution::record_host(const BaseNode       &node,
                                      const std::string &detail)
{
  this->node_executions_.push_back(
      {node.get_id(), node.get_node_type(), "host fallback", detail});
  for (int port = 0; port < node.get_nports(); ++port)
    if (node.get_port_type(port) == gngui::PortType::OUT &&
        node.get_data_type(port) == typeid(hmap::VirtualArray).name())
      if (auto *array = static_cast<hmap::VirtualArray *>(
              const_cast<BaseNode &>(node).get_data_ref(port)))
        this->fallback_tiles_ += static_cast<std::size_t>(array->get_ntiles());
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
  this->session_finished_ = true;

  // Only completed resources enter the persistent cache. The cache API keeps
  // ownership at the DeviceArray layer, so Hesiod never handles an MTLBuffer.
  if (this->cache_ && this->cache_->enabled())
    for (const auto *array : this->device_modified_)
    {
      auto it = this->device_arrays_.find(array);
      if (it != this->device_arrays_.end())
        this->cache_->store(array, it->second);
    }

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
  last_metrics_value.resident_tiles = this->resident_tiles_;
  last_metrics_value.fallback_tiles = this->fallback_tiles_;
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
  if (this->cache_)
  {
    const auto cache_stats = this->cache_->stats();
    last_metrics_value.cache_hits = cache_stats.hits;
    last_metrics_value.cache_misses = cache_stats.misses;
    last_metrics_value.cache_evictions = cache_stats.evictions;
    last_metrics_value.persistent_cache_bytes = cache_stats.bytes;
    last_metrics_value.persistent_cache_budget_bytes = cache_stats.budget_bytes;
  }
}

std::string MetalGraphExecution::diagnostics() const
{
  if (!this->enabled_ || !this->session_)
    return "backend=disabled";

  const auto stats = this->session_->stats();
  return std::format(
      "graph={} backend=Metal device={} resident_nodes={} host_nodes={} "
      "uploads={} upload_bytes={} readbacks={} readback_bytes={} "
      "command_buffers={} encoders={} synchronizations={} gpu_ms={:.3f} "
      "resident_tiles={} fallback_tiles={} cache_hits={} cache_misses={} "
      "cache_evictions={} cache_bytes={} cache_budget={}",
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
      stats.gpu_execution_ms,
      this->resident_tiles_,
      this->fallback_tiles_,
      this->cache_ ? this->cache_->stats().hits : 0,
      this->cache_ ? this->cache_->stats().misses : 0,
      this->cache_ ? this->cache_->stats().evictions : 0,
      this->cache_ ? this->cache_->stats().bytes : 0,
      this->cache_ ? this->cache_->stats().budget_bytes : 0);
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
