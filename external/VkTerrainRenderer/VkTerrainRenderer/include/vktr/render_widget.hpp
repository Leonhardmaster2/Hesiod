/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */
#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <QElapsedTimer>
#include <QTimer>
#include <QWidget>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "nlohmann/json.hpp"

// Texture name macros — compatible with QTR_TEX_* defines
#ifndef QTR_TEX_ALBEDO
#  define QTR_TEX_ALBEDO "albedo"
#  define QTR_TEX_HMAP "hmap"
#  define QTR_TEX_NORMAL "normal"
#  define QTR_TEX_SHADOW_MAP "shadow_map"
#  define QTR_TEX_DEPTH "depth"
#endif

// Forward declare to avoid pulling in full VkCompute headers
namespace vkcompute { class GpuBuffer; }

namespace vktr
{

// Matches qtr::RenderType
enum RenderType : int
{
  RENDER_2D,
  RENDER_3D
};

// ─────────────────────────────────────────────────────────────────────────────
// Uniform buffer structures — must match GLSL layout
// ─────────────────────────────────────────────────────────────────────────────

struct FrameUBO
{
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 projection;
  glm::mat4 light_space_matrix;
  glm::vec3 light_pos;      float scale_h;
  glm::vec3 camera_pos;     float hmap_h;
  glm::vec3 view_pos;       float hmap_h0;
};

struct MaterialUBO
{
  glm::vec3 base_color;     float gamma_correction;
  float shadow_strength;
  float spec_strength;
  float shininess;
  float normal_map_scaling;
  uint32_t material_flags;  // bit 0: albedo tex, bit 1: bypass shadow, bit 2: normal vis, bit 3: AO
  float ao_strength;
  float ao_radius;
  float padding0;
};

struct TerrainPushConstants
{
  int32_t  grid_width;
  int32_t  grid_height;
  float    hmap_w;
  uint32_t flags;       // bit 0: add_skirt, bit 1: wireframe
};

// ─────────────────────────────────────────────────────────────────────────────
// Swapchain resources (per-image)
// ─────────────────────────────────────────────────────────────────────────────

struct SwapchainFrame
{
  VkImage       image          = VK_NULL_HANDLE;
  VkImageView   image_view     = VK_NULL_HANDLE;
  VkFramebuffer framebuffer    = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkFence       fence          = VK_NULL_HANDLE;   // in-flight fence
};

// ─────────────────────────────────────────────────────────────────────────────
// Managed VkImage (depth, shadow map, textures)
// ─────────────────────────────────────────────────────────────────────────────

struct ManagedImage
{
  VkImage       image      = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageView   view       = VK_NULL_HANDLE;
  VkSampler     sampler    = VK_NULL_HANDLE;
  uint32_t      width = 0, height = 0;
  VkFormat      format = VK_FORMAT_UNDEFINED;

  bool valid() const { return image != VK_NULL_HANDLE; }
};

// ─────────────────────────────────────────────────────────────────────────────
// VkTerrainRenderer — drop-in replacement for qtr::RenderWidget
// ─────────────────────────────────────────────────────────────────────────────

class RenderWidget : public QWidget
{
  Q_OBJECT

public:
  explicit RenderWidget(const std::string &title = "", QWidget *parent = nullptr);
  ~RenderWidget() override;

  // --- Serialization (compatible with qtr::RenderWidget)
  void           json_from(nlohmann::json const &json);
  nlohmann::json json_to() const;

  // --- Render mode
  void set_render_type(const RenderType &new_render_type);

  // --- Visibility toggles
  bool get_bypass_texture_albedo() const { return bypass_texture_albedo_; }
  bool get_render_hmap() const { return render_hmap_; }
  bool get_render_water() const { return render_water_; }
  bool get_render_points() const { return render_points_; }
  bool get_render_path() const { return render_path_; }

  void set_bypass_texture_albedo(bool s) { bypass_texture_albedo_ = s; need_update_ = true; }
  void set_render_hmap(bool s) { render_hmap_ = s; need_update_ = true; }
  void set_render_water(bool s) { render_water_ = s; need_update_ = true; }
  void set_render_points(bool s) { render_points_ = s; need_update_ = true; }
  void set_render_path(bool s) { render_path_ = s; need_update_ = true; }
  void set_render_plane(bool) {}
  void set_render_rocks(bool) {}
  void set_render_trees(bool) {}
  void set_render_leaves(bool) {}
  bool get_render_plane() const { return true; }
  bool get_render_rocks() const { return false; }
  bool get_render_trees() const { return false; }
  bool get_render_leaves() const { return false; }

  // --- QWidget interface
  QSize sizeHint() const override { return QSize(800, 600); }

  // --- Geometry (legacy CPU path — mirrors qtr::RenderWidget API)
  void clear();

  void set_heightmap_geometry(const std::vector<float> &data,
                              int width, int height,
                              bool add_skirt = true);
  void reset_heightmap_geometry();

  // Zero-copy path: bind a GpuBuffer directly from compute pipeline
  void set_heightmap_buffer(vkcompute::GpuBuffer *buf, int width, int height);

  void set_water_geometry(const std::vector<float> &, int, int, float) {} // stub
  void reset_water_geometry() {}

  void set_points(const std::vector<float> &, const std::vector<float> &,
                  const std::vector<float> &) {} // stub
  void reset_points() {}

  void set_path(const std::vector<float> &, const std::vector<float> &,
                const std::vector<float> &) {} // stub
  void reset_path() {}

  // --- Textures (RGBA 8-bit)
  void set_texture(const std::string &name, const std::vector<uint8_t> &data, int width);
  void reset_texture(const std::string &name);
  void reset_textures();

protected:
  // --- Qt events
  void resizeEvent(QResizeEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *e) override;
  void mouseReleaseEvent(QMouseEvent *e) override;
  void mouseMoveEvent(QMouseEvent *e) override;
  void wheelEvent(QWheelEvent *e) override;

private:
  // --- Vulkan initialization
  void init_vulkan();
  void cleanup_vulkan();
  void create_surface();
  void create_swapchain();
  void cleanup_swapchain();
  void create_render_pass();
  void create_depth_resources();
  void create_framebuffers();
  void create_command_pool_and_buffers();
  void create_sync_objects();
  void create_descriptor_layouts();
  void create_pipelines();
  void create_ubo_buffers();
  void create_index_buffer();
  void create_default_textures();
  void create_descriptor_sets();

  // --- Rendering
  void render_frame();
  void record_command_buffer(VkCommandBuffer cmd, uint32_t image_index);
  void update_ubo_data();

  // --- Texture helpers
  void upload_texture(ManagedImage &img, const uint8_t *data,
                      uint32_t w, uint32_t h, VkFormat format);
  void create_managed_image(ManagedImage &img, uint32_t w, uint32_t h,
                            VkFormat format, VkImageUsageFlags usage,
                            VkImageAspectFlags aspect);
  void destroy_managed_image(ManagedImage &img);

  // --- Heightmap upload (legacy path)
  void upload_heightmap_data(const float *data, size_t count);

  // --- Pipeline helpers
  VkShaderModule create_shader_module(const std::string &spv_path);
  VkPipeline create_graphics_pipeline(const std::string &vert_spv,
                                      const std::string &frag_spv,
                                      VkPipelineLayout layout,
                                      VkRenderPass render_pass,
                                      bool depth_only = false);

  // --- Camera
  void update_camera();
  void update_light();
  void reset_camera_position();

  // ─── State ─────────────────────────────────────────────────────────────────

  std::string title_;
  RenderType  render_type_ = RenderType::RENDER_3D;
  bool        initialized_ = false;
  bool        need_update_ = true;
  bool        need_swapchain_rebuild_ = false;

  // Vulkan core
  VkSurfaceKHR  surface_ = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat       swapchain_format_ = VK_FORMAT_B8G8R8A8_SRGB;
  VkExtent2D     swapchain_extent_ = {800, 600};

  std::vector<SwapchainFrame> frames_;
  VkCommandPool command_pool_ = VK_NULL_HANDLE;

  // Sync
  static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
  VkSemaphore image_available_semaphores_[MAX_FRAMES_IN_FLIGHT] = {};
  VkSemaphore render_finished_semaphores_[MAX_FRAMES_IN_FLIGHT] = {};
  uint32_t current_frame_ = 0;

  // Render passes
  VkRenderPass main_render_pass_ = VK_NULL_HANDLE;

  // Depth buffer
  ManagedImage depth_image_;

  // Descriptor set layouts
  VkDescriptorSetLayout frame_ds_layout_ = VK_NULL_HANDLE;   // set 0: UBO + SSBO
  VkDescriptorSetLayout texture_ds_layout_ = VK_NULL_HANDLE; // set 1: samplers

  // Pipeline layouts
  VkPipelineLayout terrain_pipeline_layout_ = VK_NULL_HANDLE;

  // Pipelines
  VkPipeline terrain_pipeline_ = VK_NULL_HANDLE;

  // Descriptor pool and sets
  VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
  VkDescriptorSet  frame_descriptor_set_ = VK_NULL_HANDLE;
  VkDescriptorSet  texture_descriptor_set_ = VK_NULL_HANDLE;

  // UBO buffers (host-visible, written every frame)
  VkBuffer      frame_ubo_buffer_ = VK_NULL_HANDLE;
  VmaAllocation frame_ubo_alloc_  = VK_NULL_HANDLE;
  void         *frame_ubo_mapped_ = nullptr;

  VkBuffer      material_ubo_buffer_ = VK_NULL_HANDLE;
  VmaAllocation material_ubo_alloc_  = VK_NULL_HANDLE;
  void         *material_ubo_mapped_ = nullptr;

  // Heightmap SSBO — either owned (legacy upload) or borrowed (GpuBuffer zero-copy)
  VkBuffer      hmap_ssbo_ = VK_NULL_HANDLE;        // the buffer bound to descriptor set
  VmaAllocation hmap_ssbo_alloc_ = VK_NULL_HANDLE;   // owned allocation (legacy path only)
  vkcompute::GpuBuffer *hmap_gpu_buf_ = nullptr;     // borrowed pointer (zero-copy path)
  bool hmap_ssbo_owned_ = false;                      // true if we allocated it ourselves
  int  hmap_width_ = 0;
  int  hmap_height_ = 0;
  bool hmap_has_data_ = false;

  // Index buffer for grid (triangle list)
  VkBuffer      index_buffer_ = VK_NULL_HANDLE;
  VmaAllocation index_alloc_  = VK_NULL_HANDLE;
  uint32_t      index_count_ = 0;

  // Textures
  ManagedImage tex_albedo_;
  ManagedImage tex_hmap_;      // float heightmap texture (for AO in frag shader)
  ManagedImage tex_normal_;
  ManagedImage tex_shadow_map_;

  // Frame timer
  QTimer frame_timer_;

  // --- Visibility flags
  bool render_hmap_ = true;
  bool render_water_ = true;
  bool render_points_ = true;
  bool render_path_ = true;
  bool bypass_texture_albedo_ = false;

  // --- Camera state
  glm::vec3 cam_target_ = glm::vec3(0.f);
  glm::vec2 cam_pan_offset_ = glm::vec2(0.f);
  float     cam_distance_ = 2.5f;
  float     cam_alpha_x_ = -0.6f;    // pitch
  float     cam_alpha_y_ = 0.8f;     // yaw
  float     cam_fov_ = 45.f / 180.f * 3.14159f;
  float     cam_near_ = 0.01f;
  float     cam_far_ = 100.f;

  // --- Light state
  float light_phi_ = 0.8f;
  float light_theta_ = 1.0f;
  float light_distance_ = 10.f;

  // --- Heightmap scaling
  float scale_h_ = 1.0f;
  float hmap_h0_ = 0.f;
  float hmap_h_ = 0.4f;
  float hmap_w_ = 2.f;
  bool  add_skirt_ = true;

  // --- Rendering params
  float gamma_correction_ = 2.f;
  float shadow_strength_ = 0.9f;
  float spec_strength_ = 0.5f;
  float shininess_ = 32.f;
  float normal_map_scaling_ = 1.f;
  float ao_strength_ = 0.8f;
  float ao_radius_ = 0.03f;

  // --- Mouse interaction state
  bool              rotating_ = false;
  bool              panning_ = false;
  std::array<float, 2> last_mouse_pos_ = {0.f, 0.f};

  // --- Shader SPV path (set at init time from compile-time define or runtime)
  std::string shader_path_;
};

} // namespace vktr
