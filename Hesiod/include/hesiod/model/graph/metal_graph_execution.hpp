/* Copyright (c) 2026 Otto Link. Distributed under the terms of the GNU General
   Public License. The full license is in the file LICENSE, distributed with
   this software. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "highmap/gpu/metal.hpp"
#include "highmap/virtual_array/virtual_array.hpp"

namespace hesiod
{

class BaseNode;

struct MetalGraphNodeExecution
{
  std::string node_id;
  std::string node_type;
  std::string backend;
  std::string detail;
};

/**
 * @brief Optional bounded cache for completed graph resources.
 *
 * Entries are scoped to one GraphNode and keyed by the owning VirtualArray.
 * The cache stores HighMap DeviceArrays, never raw Metal objects or HSD data.
 * It is disabled unless HESIOD_METAL_PERSISTENT_CACHE=1 is explicitly set.
 */
class MetalGraphCache
{
public:
  struct Stats
  {
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
    std::size_t bytes = 0;
    std::size_t budget_bytes = 0;
  };

  MetalGraphCache();

  bool enabled() const noexcept { return this->enabled_; }

  hmap::gpu::metal::DeviceArray acquire(
      hmap::gpu::metal::DeviceSession       &session,
      const hmap::VirtualArray              *array);
  void store(const hmap::VirtualArray *array,
             const hmap::gpu::metal::DeviceArray &device);
  void invalidate(const hmap::VirtualArray *array);
  void clear();
  Stats stats() const noexcept { return this->stats_; }

private:
  struct Entry
  {
    hmap::gpu::metal::DeviceArray device;
    glm::ivec2                    shape = {0, 0};
    glm::ivec2                    tile_shape = {0, 0};
    int                           halo = 0;
    std::size_t                   bytes = 0;
    std::uint64_t                 last_use = 0;
  };

  void evict_until_within_budget(std::size_t required_bytes);

  bool enabled_ = false;
  std::uint64_t clock_ = 0;
  Stats stats_;
  std::unordered_map<const hmap::VirtualArray *, Entry> entries_;
};

struct MetalGraphMetrics
{
  std::string graph_id;
  std::string device;
  bool        enabled = false;
  std::size_t resident_nodes = 0;
  std::size_t host_nodes = 0;
  std::size_t host_uploads = 0;
  std::size_t host_readbacks = 0;
  std::size_t host_upload_bytes = 0;
  std::size_t host_readback_bytes = 0;
  std::size_t cache_hits = 0;
  std::size_t cache_misses = 0;
  std::size_t cache_evictions = 0;
  std::size_t persistent_cache_bytes = 0;
  std::size_t persistent_cache_budget_bytes = 0;
  std::size_t resident_tiles = 0;
  std::size_t fallback_tiles = 0;
  std::uint64_t process_rss_bytes = 0;
  std::uint64_t process_peak_rss_bytes = 0;
  std::uint64_t recommended_max_working_set_bytes = 0;
  hmap::gpu::metal::ExecutionStats metal_stats;
  std::vector<MetalGraphNodeExecution> node_executions;
};

/**
 * @brief Per-graph bridge between Hesiod VirtualArrays and Metal DeviceArrays.
 *
 * The bridge is deliberately scoped to one graph update. Device buffers are
 * never serialized into an HSD project and a host materialization is made
 * explicit whenever a non-resident node or a final consumer needs the data.
 */
class MetalGraphExecution
{
public:
  explicit MetalGraphExecution(std::string graph_id,
                               std::shared_ptr<MetalGraphCache> cache = {});
  ~MetalGraphExecution();

  MetalGraphExecution(const MetalGraphExecution &) = delete;
  MetalGraphExecution &operator=(const MetalGraphExecution &) = delete;

  static MetalGraphExecution *current();
  static const MetalGraphMetrics &last_metrics();

  bool enabled() const noexcept { return this->enabled_; }
  bool can_encode() const noexcept
  {
    return this->enabled_ && this->session_ && !this->session_finished_;
  }

  /**
   * @brief Prepare a node before its ordinary compute function runs.
   *
   * Resident-capable node implementations opt in explicitly. All other nodes
   * receive host-valid inputs at this boundary if they consume a resident
   * value, preserving the existing CPU/OpenCL implementation unchanged.
   */
  void prepare_node(BaseNode &node);
  void prepare_host_node(BaseNode       &node,
                         const std::string &detail = "host boundary");

  hmap::gpu::metal::DeviceArray device_for(const hmap::VirtualArray *array);
  void bind(const hmap::VirtualArray                 *array,
            hmap::gpu::metal::DeviceArray             device,
            bool                                      host_required);
  bool contains(const hmap::VirtualArray *array) const;
  void materialize(const hmap::VirtualArray *array);

  hmap::gpu::metal::DeviceSession &session();

  void record_resident(const BaseNode &node, const std::string &detail);
  void record_host(const BaseNode &node, const std::string &detail);

  /** @brief Materialize final graph outputs and wait for the Metal queue. */
  void flush();

  std::string diagnostics() const;
  const std::vector<MetalGraphNodeExecution> &node_executions() const
  {
    return this->node_executions_;
  }

private:
  static bool environment_enabled();
  static bool resident_candidate(const std::string &node_type);
  void        capture_metrics();

  std::string graph_id_;
  bool        enabled_ = false;
  bool        flushed_ = false;
  bool        session_finished_ = false;

  std::unique_ptr<hmap::gpu::metal::DeviceSession> session_;
  std::unordered_map<const hmap::VirtualArray *, hmap::gpu::metal::DeviceArray>
      device_arrays_;
  std::unordered_map<const hmap::VirtualArray *, bool> host_required_;
  std::unordered_set<const hmap::VirtualArray *> device_modified_;
  std::vector<MetalGraphNodeExecution>                node_executions_;
  std::shared_ptr<MetalGraphCache>                     cache_;
  std::size_t                                          host_uploads_ = 0;
  std::size_t                                          host_readbacks_ = 0;
  std::size_t                                          host_upload_bytes_ = 0;
  std::size_t                                          host_readback_bytes_ = 0;
  std::size_t                                          resident_tiles_ = 0;
  std::size_t                                          fallback_tiles_ = 0;
};

class MetalGraphExecutionScope
{
public:
  explicit MetalGraphExecutionScope(MetalGraphExecution &execution);
  ~MetalGraphExecutionScope();

  MetalGraphExecutionScope(const MetalGraphExecutionScope &) = delete;
  MetalGraphExecutionScope &operator=(const MetalGraphExecutionScope &) = delete;

private:
  MetalGraphExecution *previous_ = nullptr;
};

} // namespace hesiod
