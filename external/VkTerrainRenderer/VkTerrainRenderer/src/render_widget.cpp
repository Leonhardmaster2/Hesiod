/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General Public
   License. The full license is in the file LICENSE, distributed with this software. */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <QMouseEvent>
#include <QResizeEvent>

#include "vk_compute/device_manager.hpp"
#include "vk_compute/gpu_buffer.hpp"

#include "vktr/render_widget.hpp"

#if __has_include(<spdlog/spdlog.h>)
#  include <spdlog/spdlog.h>
#  define VKLOG(...) spdlog::info(__VA_ARGS__)
#  define VKLOG_ERR(...) spdlog::error(__VA_ARGS__)
#else
#  include <cstdio>
#  define VKLOG(msg, ...) fprintf(stdout, "[VkTR] " msg "\n", ##__VA_ARGS__)
#  define VKLOG_ERR(msg, ...) fprintf(stderr, "[VkTR ERROR] " msg "\n", ##__VA_ARGS__)
#endif

#ifndef VKTR_SHADER_PATH
#  define VKTR_SHADER_PATH ""
#endif

namespace
{

inline void vk_check(VkResult result, const char *msg)
{
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string("[VkTR] ") + msg +
                             " (VkResult=" + std::to_string(result) + ")");
}

} // anonymous namespace

namespace vktr
{

// ═════════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═════════════════════════════════════════════════════════════════════════════

RenderWidget::RenderWidget(const std::string &title, QWidget *parent)
    : QWidget(parent), title_(title)
{
  setAttribute(Qt::WA_PaintOnScreen);   // No Qt painting — we render with Vulkan
  setAttribute(Qt::WA_NativeWindow);    // Force native HWND creation
  setAttribute(Qt::WA_NoSystemBackground);
  setMinimumSize(200, 200);

  shader_path_ = VKTR_SHADER_PATH;

  reset_camera_position();

  // Deferred Vulkan init — need a valid HWND which requires the widget to be shown
  connect(&frame_timer_, &QTimer::timeout, this, [this]() {
    if (!initialized_)
    {
      if (winId() != 0)
      {
        init_vulkan();
        initialized_ = true;
      }
    }

    if (initialized_ && (need_update_ || true)) // always repaint for interactive feel
    {
      render_frame();
      need_update_ = false;
    }
  });
  frame_timer_.start(16); // ~60 FPS
}

RenderWidget::~RenderWidget()
{
  if (initialized_)
    cleanup_vulkan();
}

// ═════════════════════════════════════════════════════════════════════════════
// Vulkan Init
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::init_vulkan()
{
  VKLOG("[VkTR] Initializing Vulkan renderer...");

  create_surface();
  create_swapchain();
  create_render_pass();
  create_depth_resources();
  create_framebuffers();
  create_command_pool_and_buffers();
  create_sync_objects();
  create_descriptor_layouts();
  create_ubo_buffers();
  create_default_textures();
  create_descriptor_sets();
  create_pipelines();

  VKLOG("[VkTR] Vulkan renderer ready ({}x{})", swapchain_extent_.width, swapchain_extent_.height);
}

void RenderWidget::cleanup_vulkan()
{
  VkDevice dev = vkcompute::DeviceManager::device();
  vkDeviceWaitIdle(dev);

  // Pipelines
  if (terrain_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, terrain_pipeline_, nullptr);
  if (terrain_pipeline_layout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, terrain_pipeline_layout_, nullptr);

  // Descriptor pool (frees all sets)
  if (descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, descriptor_pool_, nullptr);

  // Descriptor set layouts
  if (frame_ds_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, frame_ds_layout_, nullptr);
  if (texture_ds_layout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, texture_ds_layout_, nullptr);

  // UBO buffers
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();
  if (frame_ubo_buffer_ != VK_NULL_HANDLE) { vmaUnmapMemory(alloc, frame_ubo_alloc_); vmaDestroyBuffer(alloc, frame_ubo_buffer_, frame_ubo_alloc_); }
  if (material_ubo_buffer_ != VK_NULL_HANDLE) { vmaUnmapMemory(alloc, material_ubo_alloc_); vmaDestroyBuffer(alloc, material_ubo_buffer_, material_ubo_alloc_); }

  // SSBO (owned)
  if (hmap_ssbo_owned_ && hmap_ssbo_ != VK_NULL_HANDLE)
    vmaDestroyBuffer(alloc, hmap_ssbo_, hmap_ssbo_alloc_);

  // Index buffer
  if (index_buffer_ != VK_NULL_HANDLE) vmaDestroyBuffer(alloc, index_buffer_, index_alloc_);

  // Textures
  destroy_managed_image(tex_albedo_);
  destroy_managed_image(tex_hmap_);
  destroy_managed_image(tex_normal_);
  destroy_managed_image(tex_shadow_map_);

  // Sync objects
  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
  {
    if (image_available_semaphores_[i] != VK_NULL_HANDLE) vkDestroySemaphore(dev, image_available_semaphores_[i], nullptr);
    if (render_finished_semaphores_[i] != VK_NULL_HANDLE) vkDestroySemaphore(dev, render_finished_semaphores_[i], nullptr);
  }

  cleanup_swapchain();

  if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(dev, command_pool_, nullptr);
  if (main_render_pass_ != VK_NULL_HANDLE) vkDestroyRenderPass(dev, main_render_pass_, nullptr);

  // Surface
  if (surface_ != VK_NULL_HANDLE)
    vkcompute::DeviceManager::destroy_surface(surface_);

  initialized_ = false;
}

// ═════════════════════════════════════════════════════════════════════════════
// Surface + Swapchain
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_surface()
{
  surface_ = vkcompute::DeviceManager::create_surface((void *)(winId()));

  if (!vkcompute::DeviceManager::queue_supports_present(surface_))
    throw std::runtime_error("[VkTR] Queue family does not support presentation");
}

void RenderWidget::create_swapchain()
{
  VkDevice dev = vkcompute::DeviceManager::device();
  VkPhysicalDevice phys = vkcompute::DeviceManager::physical_device();

  // Query surface capabilities
  VkSurfaceCapabilitiesKHR caps{};
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface_, &caps);

  // Choose extent
  if (caps.currentExtent.width != UINT32_MAX)
  {
    swapchain_extent_ = caps.currentExtent;
  }
  else
  {
    swapchain_extent_.width = std::clamp(static_cast<uint32_t>(width()),
                                         caps.minImageExtent.width,
                                         caps.maxImageExtent.width);
    swapchain_extent_.height = std::clamp(static_cast<uint32_t>(height()),
                                          caps.minImageExtent.height,
                                          caps.maxImageExtent.height);
  }

  if (swapchain_extent_.width == 0 || swapchain_extent_.height == 0)
    return; // minimized

  // Choose format
  uint32_t fmt_count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &fmt_count, nullptr);
  std::vector<VkSurfaceFormatKHR> formats(fmt_count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface_, &fmt_count, formats.data());

  swapchain_format_ = VK_FORMAT_B8G8R8A8_SRGB;
  VkColorSpaceKHR color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  for (auto &f : formats)
  {
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      swapchain_format_ = f.format;
      color_space = f.colorSpace;
      break;
    }
  }
  if (swapchain_format_ == VK_FORMAT_B8G8R8A8_SRGB && formats.size() > 0)
  {
    swapchain_format_ = formats[0].format;
    color_space = formats[0].colorSpace;
  }

  // Choose present mode (prefer FIFO = vsync)
  VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

  uint32_t image_count = caps.minImageCount + 1;
  if (caps.maxImageCount > 0 && image_count > caps.maxImageCount)
    image_count = caps.maxImageCount;

  VkSwapchainCreateInfoKHR sc_info{};
  sc_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  sc_info.surface          = surface_;
  sc_info.minImageCount    = image_count;
  sc_info.imageFormat      = swapchain_format_;
  sc_info.imageColorSpace  = color_space;
  sc_info.imageExtent      = swapchain_extent_;
  sc_info.imageArrayLayers = 1;
  sc_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sc_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sc_info.preTransform     = caps.currentTransform;
  sc_info.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sc_info.presentMode      = present_mode;
  sc_info.clipped          = VK_TRUE;
  sc_info.oldSwapchain     = VK_NULL_HANDLE;

  vk_check(vkCreateSwapchainKHR(dev, &sc_info, nullptr, &swapchain_),
           "vkCreateSwapchainKHR failed");

  // Get swapchain images
  uint32_t sc_image_count = 0;
  vkGetSwapchainImagesKHR(dev, swapchain_, &sc_image_count, nullptr);
  std::vector<VkImage> images(sc_image_count);
  vkGetSwapchainImagesKHR(dev, swapchain_, &sc_image_count, images.data());

  frames_.resize(sc_image_count);
  for (uint32_t i = 0; i < sc_image_count; i++)
  {
    frames_[i].image = images[i];

    VkImageViewCreateInfo iv_info{};
    iv_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    iv_info.image    = images[i];
    iv_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv_info.format   = swapchain_format_;
    iv_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    iv_info.subresourceRange.baseMipLevel   = 0;
    iv_info.subresourceRange.levelCount     = 1;
    iv_info.subresourceRange.baseArrayLayer = 0;
    iv_info.subresourceRange.layerCount     = 1;

    vk_check(vkCreateImageView(dev, &iv_info, nullptr, &frames_[i].image_view),
             "vkCreateImageView (swapchain) failed");
  }

  VKLOG("[VkTR] Swapchain: {}x{}, {} images", swapchain_extent_.width, swapchain_extent_.height, sc_image_count);
}

void RenderWidget::cleanup_swapchain()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  destroy_managed_image(depth_image_);

  for (auto &f : frames_)
  {
    if (f.framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, f.framebuffer, nullptr);
    if (f.image_view != VK_NULL_HANDLE)  vkDestroyImageView(dev, f.image_view, nullptr);
    if (f.fence != VK_NULL_HANDLE)       vkDestroyFence(dev, f.fence, nullptr);
  }
  frames_.clear();

  if (swapchain_ != VK_NULL_HANDLE)
  {
    vkDestroySwapchainKHR(dev, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Render Pass
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_render_pass()
{
  VkAttachmentDescription color_att{};
  color_att.format         = swapchain_format_;
  color_att.samples        = VK_SAMPLE_COUNT_1_BIT;
  color_att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  color_att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  color_att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depth_att{};
  depth_att.format         = VK_FORMAT_D32_SFLOAT;
  depth_att.samples        = VK_SAMPLE_COUNT_1_BIT;
  depth_att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_att.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_att.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference color_ref{};
  color_ref.attachment = 0;
  color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_ref{};
  depth_ref.attachment = 1;
  depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &color_ref;
  subpass.pDepthStencilAttachment = &depth_ref;

  VkSubpassDependency dep{};
  dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass    = 0;
  dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dep.srcAccessMask = 0;
  dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  VkAttachmentDescription attachments[] = {color_att, depth_att};
  VkRenderPassCreateInfo rp_info{};
  rp_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rp_info.attachmentCount = 2;
  rp_info.pAttachments    = attachments;
  rp_info.subpassCount    = 1;
  rp_info.pSubpasses      = &subpass;
  rp_info.dependencyCount = 1;
  rp_info.pDependencies   = &dep;

  vk_check(vkCreateRenderPass(vkcompute::DeviceManager::device(), &rp_info, nullptr, &main_render_pass_),
           "vkCreateRenderPass failed");
}

// ═════════════════════════════════════════════════════════════════════════════
// Depth Buffer
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_depth_resources()
{
  create_managed_image(depth_image_, swapchain_extent_.width, swapchain_extent_.height,
                       VK_FORMAT_D32_SFLOAT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT);
}

// ═════════════════════════════════════════════════════════════════════════════
// Framebuffers
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_framebuffers()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  for (auto &f : frames_)
  {
    VkImageView attachments[] = {f.image_view, depth_image_.view};

    VkFramebufferCreateInfo fb_info{};
    fb_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass      = main_render_pass_;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments    = attachments;
    fb_info.width           = swapchain_extent_.width;
    fb_info.height          = swapchain_extent_.height;
    fb_info.layers          = 1;

    vk_check(vkCreateFramebuffer(dev, &fb_info, nullptr, &f.framebuffer),
             "vkCreateFramebuffer failed");
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Command Pool + Buffers
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_command_pool_and_buffers()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = vkcompute::DeviceManager::queue_family();
  pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  vk_check(vkCreateCommandPool(dev, &pool_info, nullptr, &command_pool_),
           "vkCreateCommandPool (renderer) failed");

  // Allocate one command buffer per swapchain image
  std::vector<VkCommandBuffer> cmd_bufs(frames_.size());
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool        = command_pool_;
  alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = static_cast<uint32_t>(frames_.size());

  vk_check(vkAllocateCommandBuffers(dev, &alloc_info, cmd_bufs.data()),
           "vkAllocateCommandBuffers (renderer) failed");

  for (size_t i = 0; i < frames_.size(); i++)
    frames_[i].command_buffer = cmd_bufs[i];
}

// ═════════════════════════════════════════════════════════════════════════════
// Sync Objects
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_sync_objects()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  VkSemaphoreCreateInfo sem_info{};
  sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
  {
    vk_check(vkCreateSemaphore(dev, &sem_info, nullptr, &image_available_semaphores_[i]),
             "vkCreateSemaphore (image_available) failed");
    vk_check(vkCreateSemaphore(dev, &sem_info, nullptr, &render_finished_semaphores_[i]),
             "vkCreateSemaphore (render_finished) failed");
  }

  for (auto &f : frames_)
  {
    vk_check(vkCreateFence(dev, &fence_info, nullptr, &f.fence),
             "vkCreateFence (frame) failed");
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// Descriptor Set Layouts
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_descriptor_layouts()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  // Set 0: Frame UBO (binding 0) + Heightmap SSBO (binding 1) + Material UBO (binding 2)
  VkDescriptorSetLayoutBinding bindings_set0[3] = {};
  bindings_set0[0].binding         = 0;
  bindings_set0[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings_set0[0].descriptorCount = 1;
  bindings_set0[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  bindings_set0[1].binding         = 1;
  bindings_set0[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings_set0[1].descriptorCount = 1;
  bindings_set0[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

  bindings_set0[2].binding         = 2;
  bindings_set0[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  bindings_set0[2].descriptorCount = 1;
  bindings_set0[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layout_info0{};
  layout_info0.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info0.bindingCount = 3;
  layout_info0.pBindings    = bindings_set0;

  vk_check(vkCreateDescriptorSetLayout(dev, &layout_info0, nullptr, &frame_ds_layout_),
           "vkCreateDescriptorSetLayout (set 0) failed");

  // Set 1: Textures (4 samplers)
  VkDescriptorSetLayoutBinding bindings_set1[4] = {};
  for (int i = 0; i < 4; i++)
  {
    bindings_set1[i].binding         = static_cast<uint32_t>(i);
    bindings_set1[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings_set1[i].descriptorCount = 1;
    bindings_set1[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info1{};
  layout_info1.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info1.bindingCount = 4;
  layout_info1.pBindings    = bindings_set1;

  vk_check(vkCreateDescriptorSetLayout(dev, &layout_info1, nullptr, &texture_ds_layout_),
           "vkCreateDescriptorSetLayout (set 1) failed");
}

// ═════════════════════════════════════════════════════════════════════════════
// UBO Buffers
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_ubo_buffers()
{
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();

  auto create_ubo = [&](VkBuffer &buf, VmaAllocation &allocation, void *&mapped, size_t size)
  {
    VkBufferCreateInfo buf_info{};
    buf_info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size        = size;
    buf_info.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo result_info{};
    vk_check(static_cast<VkResult>(vmaCreateBuffer(alloc, &buf_info, &alloc_info, &buf, &allocation, &result_info)),
             "vmaCreateBuffer (UBO) failed");
    mapped = result_info.pMappedData;
  };

  create_ubo(frame_ubo_buffer_, frame_ubo_alloc_, frame_ubo_mapped_, sizeof(FrameUBO));
  create_ubo(material_ubo_buffer_, material_ubo_alloc_, material_ubo_mapped_, sizeof(MaterialUBO));
}

// ═════════════════════════════════════════════════════════════════════════════
// Managed Image
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_managed_image(ManagedImage &img, uint32_t w, uint32_t h,
                                         VkFormat format, VkImageUsageFlags usage,
                                         VkImageAspectFlags aspect)
{
  VkDevice dev = vkcompute::DeviceManager::device();
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();

  VkImageCreateInfo img_info{};
  img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_info.imageType     = VK_IMAGE_TYPE_2D;
  img_info.format        = format;
  img_info.extent.width  = w;
  img_info.extent.height = h;
  img_info.extent.depth  = 1;
  img_info.mipLevels     = 1;
  img_info.arrayLayers   = 1;
  img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
  img_info.usage         = usage;
  img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  vk_check(static_cast<VkResult>(vmaCreateImage(alloc, &img_info, &alloc_info,
                                                  &img.image, &img.allocation, nullptr)),
           "vmaCreateImage failed");

  VkImageViewCreateInfo view_info{};
  view_info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image    = img.image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format   = format;
  view_info.subresourceRange.aspectMask     = aspect;
  view_info.subresourceRange.baseMipLevel   = 0;
  view_info.subresourceRange.levelCount     = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount     = 1;

  vk_check(vkCreateImageView(dev, &view_info, nullptr, &img.view),
           "vkCreateImageView (managed) failed");

  // Create sampler if it's a sampled image
  if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
  {
    VkSamplerCreateInfo samp_info{};
    samp_info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samp_info.magFilter    = VK_FILTER_LINEAR;
    samp_info.minFilter    = VK_FILTER_LINEAR;
    samp_info.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp_info.maxLod       = 1.0f;

    vk_check(vkCreateSampler(dev, &samp_info, nullptr, &img.sampler),
             "vkCreateSampler failed");
  }

  img.width  = w;
  img.height = h;
  img.format = format;
}

void RenderWidget::destroy_managed_image(ManagedImage &img)
{
  VkDevice dev = vkcompute::DeviceManager::device();
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();

  if (img.sampler != VK_NULL_HANDLE) vkDestroySampler(dev, img.sampler, nullptr);
  if (img.view != VK_NULL_HANDLE)    vkDestroyImageView(dev, img.view, nullptr);
  if (img.image != VK_NULL_HANDLE)   vmaDestroyImage(alloc, img.image, img.allocation);

  img = ManagedImage{};
}

// ═════════════════════════════════════════════════════════════════════════════
// Default Textures (1x1 white/flat)
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_default_textures()
{
  // Create 1x1 white RGBA textures as placeholders
  uint8_t white[] = {255, 255, 255, 255};
  uint8_t flat_normal[] = {128, 128, 255, 255}; // (0,0,1) encoded as 0.5,0.5,1.0

  auto create_default = [&](ManagedImage &img, const uint8_t *data)
  {
    create_managed_image(img, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT);
    upload_texture(img, data, 1, 1, VK_FORMAT_R8G8B8A8_UNORM);
  };

  create_default(tex_albedo_, white);
  create_default(tex_normal_, flat_normal);
  create_default(tex_shadow_map_, white);

  // Heightmap texture (R32_SFLOAT) — 1x1 with value 0.5
  create_managed_image(tex_hmap_, 1, 1, VK_FORMAT_R32_SFLOAT,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);
  float half = 0.5f;
  upload_texture(tex_hmap_, reinterpret_cast<const uint8_t *>(&half), 1, 1, VK_FORMAT_R32_SFLOAT);
}

void RenderWidget::upload_texture(ManagedImage &img, const uint8_t *data,
                                   uint32_t w, uint32_t h, VkFormat format)
{
  VkDevice dev = vkcompute::DeviceManager::device();
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();

  size_t pixel_size = (format == VK_FORMAT_R32_SFLOAT) ? 4 : 4; // RGBA8 or R32F
  size_t data_size = w * h * pixel_size;

  // Create staging buffer
  VkBuffer staging_buf;
  VmaAllocation staging_alloc;

  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size  = data_size;
  buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  vmaCreateBuffer(alloc, &buf_info, &alloc_info, &staging_buf, &staging_alloc, nullptr);

  void *mapped;
  vmaMapMemory(alloc, staging_alloc, &mapped);
  memcpy(mapped, data, data_size);
  vmaUnmapMemory(alloc, staging_alloc);

  // Record copy command
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo cmd_info{};
  cmd_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_info.commandPool        = command_pool_;
  cmd_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(dev, &cmd_info, &cmd);

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

  // Transition image to TRANSFER_DST
  VkImageMemoryBarrier barrier{};
  barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.srcAccessMask       = 0;
  barrier.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.image               = img.image;
  barrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &barrier);

  VkBufferImageCopy region{};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent      = {w, h, 1};

  vkCmdCopyBufferToImage(cmd, staging_buf, img.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Transition to SHADER_READ_ONLY
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &barrier);

  vkEndCommandBuffer(cmd);

  // Submit and wait
  VkFence fence;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(dev, &fence_info, nullptr, &fence);

  VkSubmitInfo submit{};
  submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &cmd;

  {
    std::lock_guard<std::mutex> lock(vkcompute::DeviceManager::queue_mutex());
    vkQueueSubmit(vkcompute::DeviceManager::queue(), 1, &submit, fence);
  }
  vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  vkFreeCommandBuffers(dev, command_pool_, 1, &cmd);
  vmaDestroyBuffer(alloc, staging_buf, staging_alloc);
}

// ═════════════════════════════════════════════════════════════════════════════
// Descriptor Pool + Sets
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_descriptor_sets()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  // Pool
  VkDescriptorPoolSize pool_sizes[] = {
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         4},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          2},
    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
  };

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets       = 4;
  pool_info.poolSizeCount = 3;
  pool_info.pPoolSizes    = pool_sizes;

  vk_check(vkCreateDescriptorPool(dev, &pool_info, nullptr, &descriptor_pool_),
           "vkCreateDescriptorPool (renderer) failed");

  // Allocate sets
  VkDescriptorSetLayout layouts[] = {frame_ds_layout_, texture_ds_layout_};
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool     = descriptor_pool_;
  alloc_info.descriptorSetCount = 2;
  alloc_info.pSetLayouts        = layouts;

  VkDescriptorSet sets[2];
  vk_check(vkAllocateDescriptorSets(dev, &alloc_info, sets),
           "vkAllocateDescriptorSets (renderer) failed");

  frame_descriptor_set_   = sets[0];
  texture_descriptor_set_ = sets[1];

  // Write set 0: UBOs + placeholder SSBO
  VkDescriptorBufferInfo frame_ubo_info{};
  frame_ubo_info.buffer = frame_ubo_buffer_;
  frame_ubo_info.offset = 0;
  frame_ubo_info.range  = sizeof(FrameUBO);

  VkDescriptorBufferInfo material_ubo_info{};
  material_ubo_info.buffer = material_ubo_buffer_;
  material_ubo_info.offset = 0;
  material_ubo_info.range  = sizeof(MaterialUBO);

  // SSBO placeholder — will be updated when heightmap data arrives.
  // For now, bind the frame UBO as a dummy (just needs a valid buffer).
  VkDescriptorBufferInfo ssbo_info{};
  ssbo_info.buffer = frame_ubo_buffer_; // placeholder
  ssbo_info.offset = 0;
  ssbo_info.range  = sizeof(FrameUBO);

  VkWriteDescriptorSet writes_set0[3] = {};
  writes_set0[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes_set0[0].dstSet          = frame_descriptor_set_;
  writes_set0[0].dstBinding      = 0;
  writes_set0[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes_set0[0].descriptorCount = 1;
  writes_set0[0].pBufferInfo     = &frame_ubo_info;

  writes_set0[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes_set0[1].dstSet          = frame_descriptor_set_;
  writes_set0[1].dstBinding      = 1;
  writes_set0[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes_set0[1].descriptorCount = 1;
  writes_set0[1].pBufferInfo     = &ssbo_info;

  writes_set0[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes_set0[2].dstSet          = frame_descriptor_set_;
  writes_set0[2].dstBinding      = 2;
  writes_set0[2].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes_set0[2].descriptorCount = 1;
  writes_set0[2].pBufferInfo     = &material_ubo_info;

  vkUpdateDescriptorSets(dev, 3, writes_set0, 0, nullptr);

  // Write set 1: default textures
  VkDescriptorImageInfo tex_infos[4] = {};
  ManagedImage *textures[] = {&tex_albedo_, &tex_hmap_, &tex_normal_, &tex_shadow_map_};
  for (int i = 0; i < 4; i++)
  {
    tex_infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    tex_infos[i].imageView   = textures[i]->view;
    tex_infos[i].sampler     = textures[i]->sampler;
  }

  VkWriteDescriptorSet writes_set1[4] = {};
  for (int i = 0; i < 4; i++)
  {
    writes_set1[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes_set1[i].dstSet          = texture_descriptor_set_;
    writes_set1[i].dstBinding      = static_cast<uint32_t>(i);
    writes_set1[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes_set1[i].descriptorCount = 1;
    writes_set1[i].pImageInfo      = &tex_infos[i];
  }

  vkUpdateDescriptorSets(dev, 4, writes_set1, 0, nullptr);
}

// ═════════════════════════════════════════════════════════════════════════════
// Pipeline
// ═════════════════════════════════════════════════════════════════════════════

VkShaderModule RenderWidget::create_shader_module(const std::string &spv_path)
{
  std::ifstream file(spv_path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("[VkTR] Cannot open SPIR-V: " + spv_path);

  size_t size = static_cast<size_t>(file.tellg());
  if (size == 0 || size % 4 != 0)
    throw std::runtime_error("[VkTR] Invalid SPIR-V file: " + spv_path);

  std::vector<uint32_t> code(size / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(size));

  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.codeSize = size;
  info.pCode    = code.data();

  VkShaderModule mod;
  vk_check(vkCreateShaderModule(vkcompute::DeviceManager::device(), &info, nullptr, &mod),
           ("vkCreateShaderModule failed: " + spv_path).c_str());
  return mod;
}

void RenderWidget::create_pipelines()
{
  VkDevice dev = vkcompute::DeviceManager::device();

  // Push constant range
  VkPushConstantRange pc_range{};
  pc_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pc_range.offset     = 0;
  pc_range.size       = sizeof(TerrainPushConstants);

  // Pipeline layout
  VkDescriptorSetLayout set_layouts[] = {frame_ds_layout_, texture_ds_layout_};

  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount         = 2;
  layout_info.pSetLayouts            = set_layouts;
  layout_info.pushConstantRangeCount = 1;
  layout_info.pPushConstantRanges    = &pc_range;

  vk_check(vkCreatePipelineLayout(dev, &layout_info, nullptr, &terrain_pipeline_layout_),
           "vkCreatePipelineLayout (terrain) failed");

  // Load shaders
  std::string vert_path = shader_path_ + "/terrain.vert.spv";
  std::string frag_path = shader_path_ + "/terrain.frag.spv";

  VkShaderModule vert_mod = create_shader_module(vert_path);
  VkShaderModule frag_mod = create_shader_module(frag_path);

  VkPipelineShaderStageCreateInfo stages[2] = {};
  stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vert_mod;
  stages[0].pName  = "main";

  stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = frag_mod;
  stages[1].pName  = "main";

  // No vertex input (all data comes from SSBO)
  VkPipelineVertexInputStateCreateInfo vi{};
  vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo ia{};
  ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport viewport{};
  viewport.width  = static_cast<float>(swapchain_extent_.width);
  viewport.height = static_cast<float>(swapchain_extent_.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;

  VkRect2D scissor{};
  scissor.extent = swapchain_extent_;

  VkPipelineViewportStateCreateInfo vp{};
  vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vp.viewportCount = 1;
  vp.pViewports    = &viewport;
  vp.scissorCount  = 1;
  vp.pScissors     = &scissor;

  VkPipelineRasterizationStateCreateInfo rs{};
  rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode    = VK_CULL_MODE_BACK_BIT;
  rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth   = 1.0f;

  VkPipelineMultisampleStateCreateInfo ms{};
  ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo ds{};
  ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  ds.depthTestEnable  = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp   = VK_COMPARE_OP_LESS;

  VkPipelineColorBlendAttachmentState blend_att{};
  blend_att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

  VkPipelineColorBlendStateCreateInfo cb{};
  cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  cb.attachmentCount = 1;
  cb.pAttachments    = &blend_att;

  // Dynamic state for viewport/scissor on resize
  VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dyn{};
  dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dyn.dynamicStateCount = 2;
  dyn.pDynamicStates    = dynamic_states;

  VkGraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.stageCount          = 2;
  pipeline_info.pStages             = stages;
  pipeline_info.pVertexInputState   = &vi;
  pipeline_info.pInputAssemblyState = &ia;
  pipeline_info.pViewportState      = &vp;
  pipeline_info.pRasterizationState = &rs;
  pipeline_info.pMultisampleState   = &ms;
  pipeline_info.pDepthStencilState  = &ds;
  pipeline_info.pColorBlendState    = &cb;
  pipeline_info.pDynamicState       = &dyn;
  pipeline_info.layout              = terrain_pipeline_layout_;
  pipeline_info.renderPass          = main_render_pass_;
  pipeline_info.subpass             = 0;

  vk_check(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &terrain_pipeline_),
           "vkCreateGraphicsPipelines (terrain) failed");

  // Cleanup shader modules (pipeline owns the code now)
  vkDestroyShaderModule(dev, vert_mod, nullptr);
  vkDestroyShaderModule(dev, frag_mod, nullptr);

  VKLOG("[VkTR] Terrain graphics pipeline created");
}

// ═════════════════════════════════════════════════════════════════════════════
// Index Buffer
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::create_index_buffer()
{
  if (hmap_width_ < 2 || hmap_height_ < 2) return;

  VmaAllocator alloc = vkcompute::DeviceManager::allocator();

  // Free old index buffer
  if (index_buffer_ != VK_NULL_HANDLE)
  {
    vmaDestroyBuffer(alloc, index_buffer_, index_alloc_);
    index_buffer_ = VK_NULL_HANDLE;
    index_alloc_  = VK_NULL_HANDLE;
  }

  // Generate triangle indices for grid
  int w = hmap_width_;
  int h = hmap_height_;
  index_count_ = (w - 1) * (h - 1) * 6;
  std::vector<uint32_t> indices(index_count_);

  uint32_t idx = 0;
  for (int z = 0; z < h - 1; z++)
  {
    for (int x = 0; x < w - 1; x++)
    {
      uint32_t tl = z * w + x;
      uint32_t tr = tl + 1;
      uint32_t bl = (z + 1) * w + x;
      uint32_t br = bl + 1;

      indices[idx++] = tl; indices[idx++] = bl; indices[idx++] = tr;
      indices[idx++] = tr; indices[idx++] = bl; indices[idx++] = br;
    }
  }

  size_t buf_size = indices.size() * sizeof(uint32_t);

  // Create device-local index buffer with staging upload
  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size  = buf_size;
  buf_info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  vmaCreateBuffer(alloc, &buf_info, &alloc_info, &index_buffer_, &index_alloc_, nullptr);

  // Staging
  VkBuffer staging;
  VmaAllocation staging_alloc;

  VkBufferCreateInfo staging_info{};
  staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  staging_info.size  = buf_size;
  staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo staging_alloc_info{};
  staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  vmaCreateBuffer(alloc, &staging_info, &staging_alloc_info, &staging, &staging_alloc, nullptr);

  void *mapped;
  vmaMapMemory(alloc, staging_alloc, &mapped);
  memcpy(mapped, indices.data(), buf_size);
  vmaUnmapMemory(alloc, staging_alloc);

  // Copy staging → device
  VkDevice dev = vkcompute::DeviceManager::device();
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo cmd_info{};
  cmd_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_info.commandPool        = command_pool_;
  cmd_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(dev, &cmd_info, &cmd);

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

  VkBufferCopy region{};
  region.size = buf_size;
  vkCmdCopyBuffer(cmd, staging, index_buffer_, 1, &region);

  vkEndCommandBuffer(cmd);

  VkFence fence;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(dev, &fence_info, nullptr, &fence);

  VkSubmitInfo submit{};
  submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &cmd;
  {
    std::lock_guard<std::mutex> lock(vkcompute::DeviceManager::queue_mutex());
    vkQueueSubmit(vkcompute::DeviceManager::queue(), 1, &submit, fence);
  }
  vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  vkFreeCommandBuffers(dev, command_pool_, 1, &cmd);
  vmaDestroyBuffer(alloc, staging, staging_alloc);

  VKLOG("[VkTR] Index buffer: {} triangles for {}x{} grid", index_count_ / 3, w, h);
}

// ═════════════════════════════════════════════════════════════════════════════
// Heightmap Data
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::upload_heightmap_data(const float *data, size_t count)
{
  VmaAllocator alloc = vkcompute::DeviceManager::allocator();
  size_t buf_size = count * sizeof(float);

  // Re-create SSBO if needed
  if (!hmap_ssbo_owned_ || hmap_ssbo_ == VK_NULL_HANDLE || count * sizeof(float) != buf_size)
  {
    if (hmap_ssbo_owned_ && hmap_ssbo_ != VK_NULL_HANDLE)
      vmaDestroyBuffer(alloc, hmap_ssbo_, hmap_ssbo_alloc_);

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size  = buf_size;
    buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    vmaCreateBuffer(alloc, &buf_info, &alloc_info, &hmap_ssbo_, &hmap_ssbo_alloc_, nullptr);
    hmap_ssbo_owned_ = true;
  }

  // Upload via staging
  VkBuffer staging;
  VmaAllocation staging_alloc;

  VkBufferCreateInfo staging_info{};
  staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  staging_info.size  = buf_size;
  staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

  VmaAllocationCreateInfo staging_alloc_info{};
  staging_alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
  staging_alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

  vmaCreateBuffer(alloc, &staging_info, &staging_alloc_info, &staging, &staging_alloc, nullptr);

  void *mapped;
  vmaMapMemory(alloc, staging_alloc, &mapped);
  memcpy(mapped, data, buf_size);
  vmaUnmapMemory(alloc, staging_alloc);

  // Copy
  VkDevice dev = vkcompute::DeviceManager::device();
  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo cmd_alloc_info{};
  cmd_alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc_info.commandPool        = command_pool_;
  cmd_alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(dev, &cmd_alloc_info, &cmd);

  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

  VkBufferCopy region{};
  region.size = buf_size;
  vkCmdCopyBuffer(cmd, staging, hmap_ssbo_, 1, &region);

  vkEndCommandBuffer(cmd);

  VkFence fence;
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(dev, &fence_info, nullptr, &fence);

  VkSubmitInfo submit{};
  submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &cmd;
  {
    std::lock_guard<std::mutex> lock(vkcompute::DeviceManager::queue_mutex());
    vkQueueSubmit(vkcompute::DeviceManager::queue(), 1, &submit, fence);
  }
  vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(dev, fence, nullptr);
  vkFreeCommandBuffers(dev, command_pool_, 1, &cmd);
  vmaDestroyBuffer(alloc, staging, staging_alloc);

  // Update descriptor set binding 1 (SSBO)
  VkDescriptorBufferInfo ssbo_info{};
  ssbo_info.buffer = hmap_ssbo_;
  ssbo_info.offset = 0;
  ssbo_info.range  = buf_size;

  VkWriteDescriptorSet write{};
  write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet          = frame_descriptor_set_;
  write.dstBinding      = 1;
  write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo     = &ssbo_info;

  vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
}

void RenderWidget::set_heightmap_geometry(const std::vector<float> &data,
                                           int width, int height,
                                           bool add_skirt)
{
  if (!initialized_) return;
  if (data.empty() || width < 2 || height < 2) return;

  vkDeviceWaitIdle(vkcompute::DeviceManager::device());

  hmap_width_  = width;
  hmap_height_ = height;
  add_skirt_   = add_skirt;
  hmap_gpu_buf_ = nullptr; // not using zero-copy

  upload_heightmap_data(data.data(), data.size());
  create_index_buffer();

  // Also upload heightmap as a float texture (for AO in fragment shader)
  destroy_managed_image(tex_hmap_);
  create_managed_image(tex_hmap_, width, height, VK_FORMAT_R32_SFLOAT,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);
  upload_texture(tex_hmap_, reinterpret_cast<const uint8_t *>(data.data()),
                 width, height, VK_FORMAT_R32_SFLOAT);

  // Update texture descriptor for hmap
  VkDescriptorImageInfo img_info{};
  img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  img_info.imageView   = tex_hmap_.view;
  img_info.sampler     = tex_hmap_.sampler;

  VkWriteDescriptorSet write{};
  write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet          = texture_descriptor_set_;
  write.dstBinding      = 1;
  write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.descriptorCount = 1;
  write.pImageInfo      = &img_info;

  vkUpdateDescriptorSets(vkcompute::DeviceManager::device(), 1, &write, 0, nullptr);

  hmap_has_data_ = true;
  need_update_ = true;

  VKLOG("[VkTR] Heightmap uploaded: {}x{} ({} floats)", width, height, data.size());
}

void RenderWidget::set_heightmap_buffer(vkcompute::GpuBuffer *buf, int width, int height)
{
  if (!initialized_ || !buf) return;

  vkDeviceWaitIdle(vkcompute::DeviceManager::device());

  hmap_width_  = width;
  hmap_height_ = height;
  hmap_gpu_buf_ = buf;

  // Free owned SSBO if we had one
  if (hmap_ssbo_owned_ && hmap_ssbo_ != VK_NULL_HANDLE)
  {
    vmaDestroyBuffer(vkcompute::DeviceManager::allocator(), hmap_ssbo_, hmap_ssbo_alloc_);
    hmap_ssbo_owned_ = false;
  }

  // Bind the GpuBuffer's VkBuffer directly — zero copy!
  hmap_ssbo_ = buf->vk_buffer();
  hmap_ssbo_alloc_ = VK_NULL_HANDLE;

  // Update descriptor set
  VkDescriptorBufferInfo ssbo_info{};
  ssbo_info.buffer = hmap_ssbo_;
  ssbo_info.offset = 0;
  ssbo_info.range  = buf->size_bytes();

  VkWriteDescriptorSet write{};
  write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet          = frame_descriptor_set_;
  write.dstBinding      = 1;
  write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.descriptorCount = 1;
  write.pBufferInfo     = &ssbo_info;

  vkUpdateDescriptorSets(vkcompute::DeviceManager::device(), 1, &write, 0, nullptr);

  create_index_buffer();
  hmap_has_data_ = true;
  need_update_ = true;

  VKLOG("[VkTR] Heightmap bound zero-copy: {}x{}, GpuBuffer={}", width, height, (void*)buf);
}

void RenderWidget::reset_heightmap_geometry()
{
  hmap_has_data_ = false;
  need_update_ = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Textures
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::set_texture(const std::string &name, const std::vector<uint8_t> &data, int width)
{
  if (!initialized_) return;
  if (data.empty() || width <= 0) return;

  vkDeviceWaitIdle(vkcompute::DeviceManager::device());

  int height = static_cast<int>(data.size()) / (width * 4); // RGBA
  if (height <= 0) return;

  ManagedImage *target = nullptr;
  int binding = -1;

  if (name == "albedo")      { target = &tex_albedo_;     binding = 0; }
  else if (name == "normal") { target = &tex_normal_;     binding = 2; }
  // hmap and shadow_map are managed internally

  if (!target || binding < 0) return;

  destroy_managed_image(*target);
  create_managed_image(*target, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);
  upload_texture(*target, data.data(), width, height, VK_FORMAT_R8G8B8A8_UNORM);

  // Update descriptor
  VkDescriptorImageInfo img_info{};
  img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  img_info.imageView   = target->view;
  img_info.sampler     = target->sampler;

  VkWriteDescriptorSet write{};
  write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet          = texture_descriptor_set_;
  write.dstBinding      = static_cast<uint32_t>(binding);
  write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  write.descriptorCount = 1;
  write.pImageInfo      = &img_info;

  vkUpdateDescriptorSets(vkcompute::DeviceManager::device(), 1, &write, 0, nullptr);
  need_update_ = true;
}

void RenderWidget::reset_texture(const std::string &name)
{
  // Reset to default 1x1 textures
  if (name == "albedo")
  {
    destroy_managed_image(tex_albedo_);
    uint8_t white[] = {255, 255, 255, 255};
    create_managed_image(tex_albedo_, 1, 1, VK_FORMAT_R8G8B8A8_UNORM,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT);
    upload_texture(tex_albedo_, white, 1, 1, VK_FORMAT_R8G8B8A8_UNORM);
  }
  need_update_ = true;
}

void RenderWidget::reset_textures()
{
  reset_texture("albedo");
  reset_texture("normal");
}

void RenderWidget::clear()
{
  hmap_has_data_ = false;
  reset_textures();
  need_update_ = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Camera / Light
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::reset_camera_position()
{
  cam_target_    = glm::vec3(0.f);
  cam_distance_  = 2.5f;
  cam_alpha_x_   = -0.6f;
  cam_alpha_y_   = 0.8f;
  cam_pan_offset_= glm::vec2(0.f);
}

// ═════════════════════════════════════════════════════════════════════════════
// UBO Update
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::update_ubo_data()
{
  float aspect = static_cast<float>(swapchain_extent_.width) /
                 std::max(1.0f, static_cast<float>(swapchain_extent_.height));

  // Camera position from spherical coordinates
  float cx = cam_distance_ * cos(cam_alpha_x_) * sin(cam_alpha_y_);
  float cy = cam_distance_ * sin(cam_alpha_x_);
  float cz = cam_distance_ * cos(cam_alpha_x_) * cos(cam_alpha_y_);
  glm::vec3 cam_pos = cam_target_ + glm::vec3(cx, cy, cz);

  // Light position from spherical coordinates
  float lx = light_distance_ * sin(light_theta_) * cos(light_phi_);
  float ly = light_distance_ * cos(light_theta_);
  float lz = light_distance_ * sin(light_theta_) * sin(light_phi_);
  glm::vec3 light_pos = glm::vec3(lx, ly, lz);

  glm::mat4 view_mat = glm::lookAt(cam_pos, cam_target_, glm::vec3(0.f, 1.f, 0.f));
  glm::mat4 proj_mat = glm::perspective(cam_fov_, aspect, cam_near_, cam_far_);

  // Vulkan clip space correction: Y is inverted, depth [0,1]
  proj_mat[1][1] *= -1.0f;

  // Light space matrix for shadow mapping
  float ortho_size = 2.f;
  glm::mat4 light_proj = glm::ortho(-ortho_size, ortho_size, -ortho_size, ortho_size,
                                     0.1f, light_distance_ * 2.f);
  light_proj[1][1] *= -1.0f;
  glm::mat4 light_view = glm::lookAt(light_pos, glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));

  // Frame UBO
  FrameUBO frame{};
  frame.model              = glm::mat4(1.0f);
  frame.view               = view_mat;
  frame.projection         = proj_mat;
  frame.light_space_matrix = light_proj * light_view;
  frame.light_pos          = light_pos;
  frame.scale_h            = scale_h_;
  frame.camera_pos         = cam_pos;
  frame.hmap_h             = hmap_h_;
  frame.view_pos           = cam_pos;
  frame.hmap_h0            = hmap_h0_;

  memcpy(frame_ubo_mapped_, &frame, sizeof(FrameUBO));

  // Material UBO
  MaterialUBO mat{};
  mat.base_color        = glm::vec3(0.7f, 0.7f, 0.7f);
  mat.gamma_correction  = gamma_correction_;
  mat.shadow_strength   = shadow_strength_;
  mat.spec_strength     = spec_strength_;
  mat.shininess         = shininess_;
  mat.normal_map_scaling= normal_map_scaling_;
  mat.material_flags    = 0;
  if (!bypass_texture_albedo_) mat.material_flags |= 1u;
  // bit 1: bypass_shadow — shadow map not implemented yet, always bypass
  mat.material_flags |= 2u;
  // bit 3: AO
  mat.material_flags |= 8u;
  mat.ao_strength       = ao_strength_;
  mat.ao_radius         = ao_radius_;

  memcpy(material_ubo_mapped_, &mat, sizeof(MaterialUBO));
}

// ═════════════════════════════════════════════════════════════════════════════
// Command Buffer Recording
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::record_command_buffer(VkCommandBuffer cmd, uint32_t image_index)
{
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd, &begin);

  VkClearValue clear_values[2] = {};
  clear_values[0].color = {{0.15f, 0.15f, 0.18f, 1.0f}};
  clear_values[1].depthStencil = {1.0f, 0};

  VkRenderPassBeginInfo rp_begin{};
  rp_begin.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  rp_begin.renderPass      = main_render_pass_;
  rp_begin.framebuffer     = frames_[image_index].framebuffer;
  rp_begin.renderArea      = {{0, 0}, swapchain_extent_};
  rp_begin.clearValueCount = 2;
  rp_begin.pClearValues    = clear_values;

  vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

  // Set dynamic viewport/scissor
  VkViewport viewport{};
  viewport.width    = static_cast<float>(swapchain_extent_.width);
  viewport.height   = static_cast<float>(swapchain_extent_.height);
  viewport.minDepth = 0.f;
  viewport.maxDepth = 1.f;
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  VkRect2D scissor{};
  scissor.extent = swapchain_extent_;
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  // Draw terrain if we have data
  if (hmap_has_data_ && render_hmap_ && index_count_ > 0)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, terrain_pipeline_);

    VkDescriptorSet sets[] = {frame_descriptor_set_, texture_descriptor_set_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            terrain_pipeline_layout_, 0, 2, sets, 0, nullptr);

    TerrainPushConstants pc{};
    pc.grid_width  = hmap_width_;
    pc.grid_height = hmap_height_;
    pc.hmap_w      = hmap_w_;
    pc.flags       = add_skirt_ ? 1u : 0u;

    vkCmdPushConstants(cmd, terrain_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(TerrainPushConstants), &pc);

    vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, index_count_, 1, 0, 0, 0);
  }

  vkCmdEndRenderPass(cmd);
  vkEndCommandBuffer(cmd);
}

// ═════════════════════════════════════════════════════════════════════════════
// Frame Rendering
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::render_frame()
{
  if (!initialized_ || swapchain_ == VK_NULL_HANDLE) return;
  if (swapchain_extent_.width == 0 || swapchain_extent_.height == 0) return;

  VkDevice dev = vkcompute::DeviceManager::device();

  // Update UBO data
  update_ubo_data();

  // Acquire next swapchain image
  uint32_t image_index;
  VkResult result = vkAcquireNextImageKHR(dev, swapchain_, UINT64_MAX,
                                           image_available_semaphores_[current_frame_],
                                           VK_NULL_HANDLE, &image_index);

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
  {
    // Recreate swapchain
    vkDeviceWaitIdle(dev);
    cleanup_swapchain();
    create_swapchain();
    create_depth_resources();
    create_framebuffers();

    // Re-allocate command buffers
    std::vector<VkCommandBuffer> cmd_bufs(frames_.size());
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = command_pool_;
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(frames_.size());
    vkAllocateCommandBuffers(dev, &alloc_info, cmd_bufs.data());
    for (size_t i = 0; i < frames_.size(); i++)
      frames_[i].command_buffer = cmd_bufs[i];

    // Re-create fences
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto &f : frames_)
      vkCreateFence(dev, &fence_info, nullptr, &f.fence);

    return; // retry next frame
  }
  if (result != VK_SUCCESS) return;

  // Wait for this frame's previous work to complete
  vkWaitForFences(dev, 1, &frames_[image_index].fence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &frames_[image_index].fence);

  // Record command buffer
  vkResetCommandBuffer(frames_[image_index].command_buffer, 0);
  record_command_buffer(frames_[image_index].command_buffer, image_index);

  // Submit
  VkSemaphore wait_sems[]   = {image_available_semaphores_[current_frame_]};
  VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  VkSemaphore signal_sems[] = {render_finished_semaphores_[current_frame_]};

  VkSubmitInfo submit{};
  submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.waitSemaphoreCount   = 1;
  submit.pWaitSemaphores      = wait_sems;
  submit.pWaitDstStageMask    = wait_stages;
  submit.commandBufferCount   = 1;
  submit.pCommandBuffers      = &frames_[image_index].command_buffer;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores    = signal_sems;

  {
    std::lock_guard<std::mutex> lock(vkcompute::DeviceManager::queue_mutex());
    vk_check(vkQueueSubmit(vkcompute::DeviceManager::queue(), 1, &submit, frames_[image_index].fence),
             "vkQueueSubmit (render) failed");
  }

  // Present
  VkPresentInfoKHR present{};
  present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores    = signal_sems;
  present.swapchainCount     = 1;
  present.pSwapchains        = &swapchain_;
  present.pImageIndices      = &image_index;

  {
    std::lock_guard<std::mutex> lock(vkcompute::DeviceManager::queue_mutex());
    result = vkQueuePresentKHR(vkcompute::DeviceManager::queue(), &present);
  }

  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    need_swapchain_rebuild_ = true;

  current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ═════════════════════════════════════════════════════════════════════════════
// Qt Events
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::paintEvent(QPaintEvent *)
{
  // Rendering is driven by the frame timer, not paintEvent
}

void RenderWidget::resizeEvent(QResizeEvent *)
{
  if (!initialized_) return;

  need_swapchain_rebuild_ = true;

  // Rebuild swapchain on next frame
  VkDevice dev = vkcompute::DeviceManager::device();
  vkDeviceWaitIdle(dev);
  cleanup_swapchain();
  create_swapchain();

  if (swapchain_extent_.width > 0 && swapchain_extent_.height > 0)
  {
    create_depth_resources();
    create_framebuffers();

    // Re-allocate command buffers
    std::vector<VkCommandBuffer> cmd_bufs(frames_.size());
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool        = command_pool_;
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = static_cast<uint32_t>(frames_.size());
    vkAllocateCommandBuffers(dev, &alloc_info, cmd_bufs.data());
    for (size_t i = 0; i < frames_.size(); i++)
      frames_[i].command_buffer = cmd_bufs[i];

    // Re-create fences
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto &f : frames_)
      vkCreateFence(dev, &fence_info, nullptr, &f.fence);
  }

  need_swapchain_rebuild_ = false;
  need_update_ = true;
}

void RenderWidget::mousePressEvent(QMouseEvent *e)
{
  if (e->button() == Qt::LeftButton)
  {
    rotating_ = true;
    last_mouse_pos_ = {static_cast<float>(e->pos().x()), static_cast<float>(e->pos().y())};
  }
  else if (e->button() == Qt::RightButton)
  {
    panning_ = true;
    last_mouse_pos_ = {static_cast<float>(e->pos().x()), static_cast<float>(e->pos().y())};
  }
}

void RenderWidget::mouseReleaseEvent(QMouseEvent *e)
{
  if (e->button() == Qt::LeftButton)  rotating_ = false;
  if (e->button() == Qt::RightButton) panning_ = false;
}

void RenderWidget::mouseMoveEvent(QMouseEvent *e)
{
  float x = static_cast<float>(e->pos().x());
  float y = static_cast<float>(e->pos().y());

  float dx = x - last_mouse_pos_[0];
  float dy = y - last_mouse_pos_[1];

  if (rotating_)
  {
    cam_alpha_y_ += dx * 0.005f;
    cam_alpha_x_ -= dy * 0.005f;
    cam_alpha_x_ = std::clamp(cam_alpha_x_, -1.5f, 1.5f);
    need_update_ = true;
  }

  if (panning_)
  {
    cam_target_.x -= dx * 0.002f * cam_distance_;
    cam_target_.z -= dy * 0.002f * cam_distance_;
    need_update_ = true;
  }

  last_mouse_pos_ = {x, y};
}

void RenderWidget::wheelEvent(QWheelEvent *e)
{
  float delta = static_cast<float>(e->angleDelta().y()) / 120.f;
  cam_distance_ *= (1.f - 0.1f * delta);
  cam_distance_ = std::clamp(cam_distance_, 0.1f, 50.f);
  need_update_ = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// Serialization
// ═════════════════════════════════════════════════════════════════════════════

void RenderWidget::json_from(nlohmann::json const &json)
{
  if (json.contains("cam_distance"))  cam_distance_ = json["cam_distance"];
  if (json.contains("cam_alpha_x"))   cam_alpha_x_ = json["cam_alpha_x"];
  if (json.contains("cam_alpha_y"))   cam_alpha_y_ = json["cam_alpha_y"];
  if (json.contains("light_phi"))     light_phi_ = json["light_phi"];
  if (json.contains("light_theta"))   light_theta_ = json["light_theta"];
  if (json.contains("scale_h"))       scale_h_ = json["scale_h"];
  if (json.contains("render_type"))   render_type_ = static_cast<RenderType>(json["render_type"].get<int>());
  need_update_ = true;
}

nlohmann::json RenderWidget::json_to() const
{
  nlohmann::json json;
  json["cam_distance"] = cam_distance_;
  json["cam_alpha_x"]  = cam_alpha_x_;
  json["cam_alpha_y"]  = cam_alpha_y_;
  json["light_phi"]    = light_phi_;
  json["light_theta"]  = light_theta_;
  json["scale_h"]      = scale_h_;
  json["render_type"]  = static_cast<int>(render_type_);
  return json;
}

void RenderWidget::set_render_type(const RenderType &new_render_type)
{
  render_type_ = new_render_type;
  need_update_ = true;
}

} // namespace vktr
