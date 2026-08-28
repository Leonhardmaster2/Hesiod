/* Copyright (c) 2025 Otto Link. Distributed under the terms of the GNU General
 * Public License. The full license is in the file LICENSE, distributed with
 * this software. */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>

#include <QTimer>

#include "hesiod/app/hesiod_application.hpp"
#include "hesiod/cli/batch_mode.hpp"
#include "hesiod/gui/project_ui.hpp"
#include "hesiod/gui/widgets/graph_tabs_widget.hpp"
#include "hesiod/gui/widgets/gui_utils.hpp"
#include "hesiod/logger.hpp"
#include "hesiod/model/graph/graph_manager.hpp"
#include "hesiod/model/graph/graph_node.hpp"
#include "hesiod/model/graph/metal_graph_execution.hpp"
#include "hesiod/model/nodes/base_node.hpp"
#include "hesiod/model/nodes/node_factory.hpp"
#include "hesiod/model/nodes/post_process.hpp"

namespace hesiod::cli
{

int parse_args(args::ArgumentParser &parser,
               int                   argc,
               char                 *argv[],
               std::string          &startup_file)
{
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

  args::ValueFlag<std::string> file_flag(parser,
                                         "hsd file",
                                         "Project file (.hsd) to open at startup",
                                         {'f', "file"});

  args::Positional<std::string> file_positional(parser,
                                                "file",
                                                "Project file (.hsd) to open at startup");

  args::Group group(parser,
                    "This group is all exclusive:",
                    args::Group::Validators::DontCare);

  args::Flag snapshot_generation(group, "", "Node snapshot generation", {"snapshot"});

  args::Flag node_inventory(group, "", "Node inventory output", {"inventory"});

  args::Flag check_port_links(group,
                              "check-port-links",
                              "verify drag-to-create port rules",
                              {"check-port-links"});

  args::ValueFlag<std::string> batch(group,
                                     "hsd file",
                                     "Execute Hesiod in batch mode",
                                     {'b', "batch"});

  args::ValueFlag<std::string> phase4_benchmark(
      group,
      "hsd file",
      "Run the Phase 4 resident-vs-fallback graph benchmark",
      {"phase4-benchmark"});

  args::Group batch_args(group,
                         "batch mode arguments",
                         args::Group::Validators::DontCare);

  args::ValueFlag<glm::ivec2> shape_arg(
      batch_args,
      "shape",
      "Heightmap shape (in pixels), ex. --shape=512,512",
      {"shape"});

  args::ValueFlag<glm::ivec2> tiling_arg(batch_args,
                                         "tiling",
                                         "Heightmap tiling, ex. --tiling=4,4",
                                         {"tiling"});

  args::ValueFlag<float> overlap_arg(
      batch_args,
      "overlap",
      "Tile overlapping ratio (in [0, 1[), ex. --overlap=0.25",
      {"overlap"});

  try
  {
    parser.ParseCLI(argc, argv);

    if (phase4_benchmark)
    {
      run_phase4_benchmark(args::get(phase4_benchmark),
                           shape_arg ? args::get(shape_arg) : glm::ivec2(0, 0),
                           tiling_arg ? args::get(tiling_arg) : glm::ivec2(0, 0),
                           overlap_arg ? args::get(overlap_arg) : -1.f);
      return 0;
    }
    else if (batch)
    {
      run_batch_mode(args::get(batch),
                     shape_arg ? args::get(shape_arg) : glm::ivec2(0, 0),
                     tiling_arg ? args::get(tiling_arg) : glm::ivec2(0, 0),
                     overlap_arg ? args::get(overlap_arg) : -1.f);
      return 0;
    }
    else if (snapshot_generation)
    {
      run_snapshot_generation();
      return 0;
    }
    else if (node_inventory)
    {
      run_node_inventory();
      return 0;
    }
    else if (check_port_links)
    {
      return hesiod::run_check_port_links();
    }

    if (file_flag && file_positional &&
        args::get(file_flag) != args::get(file_positional))
    {
      std::cerr << "Error: conflicting project files requested: positional argument is "
                << "'" << args::get(file_positional) << "' but -f/--file is " << "'"
                << args::get(file_flag) << "'. "
                << "Provide only one of them (or make them identical)." << std::endl;
      return 1;
    }

    if (file_flag)
      startup_file = args::get(file_flag);
    else if (file_positional)
      startup_file = args::get(file_positional);
  }
  catch (const args::Help &help)
  {
    std::cout << parser;
    return 0;
  }

  catch (args::Error &e)
  {
    std::cerr << e.what() << std::endl << parser;
    return 1;
  }

  return -1;
}

void run_batch_mode(const std::string &filename,
                    const glm::ivec2  &shape,
                    const glm::ivec2  &tiling,
                    float              overlap,
                    const GraphConfig *p_input_model_config)
{
  Logger::log()->info("executing Hesiod in batch mode");
  Logger::log()->trace("file: {}", filename);
  Logger::log()->trace("cli shape: {{{}, {}}}", shape.x, shape.y);
  Logger::log()->trace("cli tiling: {{{}, {}}}", tiling.x, tiling.y);
  Logger::log()->trace("cli overlap: {}", overlap);

  // define actual computation configuration based on CLI inputs. If
  // nothing is provided, use the configs from the input file but if
  // an input config is provided, this config is used for all the
  // graph nodes.
  hesiod::GraphConfig config;

  // override some parameters on request
  if (p_input_model_config)
  {
    config.cm_cpu.mode = p_input_model_config->cm_cpu.mode;
    config.cm_gpu.mode = p_input_model_config->cm_gpu.mode;

    // force memory release after each node computation
    config.cm_cpu.trim_storage = true;
    config.cm_gpu.trim_storage = true;
    config.cm_single_array.trim_storage = true;
  }

  if (shape.x || shape.y || tiling.x || tiling.y || overlap >= 0.f)
  {
    glm::ivec2 new_shape = (shape.x && shape.y) ? shape : glm::ivec2(1024, 1024);
    glm::ivec2 new_tiling = (tiling.x && tiling.y) ? tiling : glm::ivec2(1, 1);
    float      new_overlap = overlap >= 0.f
                                 ? overlap
                                 : ((config.tiling.x == 1 && config.tiling.y == 1) ? 0.f
                                                                                   : 0.5f);

    config.set_shape(new_shape);
    config.set_tiling(new_tiling);
    config.set_overlap(new_overlap);

    Logger::log()->info("graph configurations will be overriden:");
    Logger::log()->info("compute shape: {{{}, {}}}", config.shape.x, config.shape.y);
    Logger::log()->info("compute tiling: {{{}, {}}}", config.tiling.x, config.tiling.y);
    Logger::log()->info("compute overlap: {}", config.overlap);
  }

  GraphManager graph_manager;
  graph_manager.load_from_file(filename, &config);

  // flatten & export if there is a configuration defined
  if (!graph_manager.get_export_param().export_path.empty())
    graph_manager.export_flatten();
}

namespace
{

struct Phase4Run
{
  double                         wall_ms = 0.;
  double                         preview_ms = 0.;
  hmap::Array                    output;
  bool                           has_output = false;
  MetalGraphMetrics              metrics;
  std::vector<std::pair<std::string, NodeRuntimeInfo>> nodes;
};

struct Phase4Output
{
  hmap::Array data;
  std::string node_id;
  std::string port;
};

std::optional<Phase4Output> phase4_output(GraphManager &manager,
                                          const GraphConfig &config)
{
  std::optional<Phase4Output> fallback;

  for (const auto &graph_id : manager.get_graph_order())
  {
    auto *graph = manager.get_graph_ref_by_id(graph_id);
    if (!graph)
      continue;

    for (const auto &[node_id, node_ptr] : graph->get_nodes())
    {
      auto *node = dynamic_cast<BaseNode *>(node_ptr.get());
      if (!node)
        continue;

      for (int port = 0; port < node->get_nports(); ++port)
      {
        if (node->get_port_type(port) != gngui::PortType::OUT ||
            node->get_data_type(port) != typeid(hmap::VirtualArray).name())
          continue;

        auto *array = node->get_value_ref<hmap::VirtualArray>(port);
        if (!array)
          continue;

        Phase4Output candidate{array->to_array(config.cm_single_array),
                               node_id,
                               node->get_port_label(port)};

        // SpectralEqualizer.hsd ends at Blend(10). Prefer that terminal
        // output, while keeping the helper useful for other small graphs.
        if (node->get_node_type() == "Blend")
          return candidate;
        fallback = std::move(candidate);
      }
    }
  }

  return fallback;
}

Phase4Run phase4_evaluate(const std::string &filename,
                          GraphConfig        config,
                          bool               resident)
{
  ::setenv("HESIOD_METAL_RESIDENT", resident ? "1" : "0", 1);

  GraphManager manager;
  const auto start = std::chrono::steady_clock::now();
  manager.load_from_file(filename, &config);
  const auto end = std::chrono::steady_clock::now();

  Phase4Run result;
  result.wall_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.metrics = MetalGraphExecution::last_metrics();

  for (const auto &graph_id : manager.get_graph_order())
  {
    auto *graph = manager.get_graph_ref_by_id(graph_id);
    if (!graph)
      continue;
    for (const auto &[node_id, node_ptr] : graph->get_nodes())
    {
      if (auto *node = dynamic_cast<BaseNode *>(node_ptr.get()))
        result.nodes.emplace_back(node_id, node->get_runtime_info());
    }
  }

  const auto preview_start = std::chrono::steady_clock::now();
  if (auto output = phase4_output(manager, config))
  {
    result.output = std::move(output->data);
    result.has_output = true;
  }
  const auto preview_end = std::chrono::steady_clock::now();
  result.preview_ms =
      std::chrono::duration<double, std::milli>(preview_end - preview_start).count();

  return result;
}

double max_abs_difference(const hmap::Array &a, const hmap::Array &b)
{
  if (a.shape != b.shape || a.vector.size() != b.vector.size())
    return std::numeric_limits<double>::max();

  double max_abs = 0.;
  for (size_t i = 0; i < a.vector.size(); ++i)
    max_abs = std::max(max_abs, std::abs((double)a.vector[i] - (double)b.vector[i]));
  return max_abs;
}

void print_phase4_run(const char *label, const Phase4Run &run)
{
  const auto &m = run.metrics;
  std::cout << std::fixed << std::setprecision(3)
            << "PHASE4 mode=" << label << " wall_ms=" << run.wall_ms
            << " preview_ms=" << run.preview_ms << " resident_nodes="
            << m.resident_nodes << " host_nodes=" << m.host_nodes
            << " uploads=" << m.host_uploads << " upload_bytes=" << m.host_upload_bytes
            << " readbacks=" << m.host_readbacks
            << " readback_bytes=" << m.host_readback_bytes
            << " command_buffers=" << m.metal_stats.command_buffers
            << " encoders=" << m.metal_stats.encoders
            << " synchronizations=" << m.metal_stats.synchronization_count
            << " gpu_ms=" << m.metal_stats.gpu_execution_ms << '\n';

  for (const auto &[node_id, info] : run.nodes)
    std::cout << std::fixed << std::setprecision(3) << "PHASE4_NODE mode=" << label
              << " id=" << node_id << " update_ms=" << info.update_time
              << " evals=" << info.eval_count << " errors=" << info.error_count << '\n';
}

} // namespace

void run_phase4_benchmark(const std::string &filename,
                          const glm::ivec2  &shape,
                          const glm::ivec2  &tiling,
                          float              overlap)
{
  Logger::log()->info("executing Phase 4 resident graph benchmark");

  GraphConfig config;
  const glm::ivec2 benchmark_shape = (shape.x && shape.y) ? shape : glm::ivec2(512, 512);
  const glm::ivec2 benchmark_tiling = (tiling.x && tiling.y) ? tiling : glm::ivec2(1, 1);
  const float benchmark_overlap = overlap >= 0.f ? overlap : 0.f;
  config.set_shape(benchmark_shape);
  config.set_tiling(benchmark_tiling);
  config.set_overlap(benchmark_overlap);

  const Phase4Run fallback = phase4_evaluate(filename, config, false);
  const Phase4Run resident = phase4_evaluate(filename, config, true);
  constexpr double phase4_parity_tolerance = 1e-2;

  print_phase4_run("fallback", fallback);
  print_phase4_run("resident", resident);

  if (fallback.has_output && resident.has_output)
  {
    std::cout << std::fixed << std::setprecision(8)
              << "PHASE4_PARITY shape=" << benchmark_shape.x << "x" << benchmark_shape.y
              << " max_abs=" << max_abs_difference(fallback.output, resident.output)
              << " status="
              << (max_abs_difference(fallback.output, resident.output) <=
                          phase4_parity_tolerance
                      ? "PASS"
                      : "FAIL")
              << '\n';
  }

  // Exercise dirty-node propagation on the real branch point. SpectralEqualizer
  // remains cached; changing Thermal(11) must update Thermal and its Blend(10)
  // consumer without rebuilding the unrelated branch.
  ::setenv("HESIOD_METAL_RESIDENT", "1", 1);
  GraphManager edited;
  edited.load_from_file(filename, &config);
  if (!edited.get_graph_order().empty())
  {
    auto *graph = edited.get_graph_ref_by_id(edited.get_graph_order().front());
    auto *thermal = graph ? graph->get_node_ref_by_id<BaseNode>("11") : nullptr;
    if (thermal)
    {
      thermal->set_value<float>("duration", thermal->val<float>("duration") * 0.5f);
      const auto edit_start = std::chrono::steady_clock::now();
      graph->update("11");
      const auto edit_end = std::chrono::steady_clock::now();
      const double edit_ms =
          std::chrono::duration<double, std::milli>(edit_end - edit_start).count();
      const auto metrics = MetalGraphExecution::last_metrics();
      std::cout << std::fixed << std::setprecision(3) << "PHASE4_EDIT wall_ms=" << edit_ms
                << " resident_nodes=" << metrics.resident_nodes
                << " host_nodes=" << metrics.host_nodes << '\n';
    }
  }

  ::unsetenv("HESIOD_METAL_RESIDENT");
}

void run_node_inventory()
{
  Logger::log()->info("executing Hesiod in node inventory mode");
  hesiod::dump_node_inventory("node_inventory");

  auto config = std::make_shared<hesiod::GraphConfig>();
  hesiod::dump_node_documentation_stub("node_documentation_stub.json", config);

  hesiod::dump_node_settings_screenshots();
}

void run_snapshot_generation()
{
  Logger::log()->info("executing Hesiod in snapshot generation mode");

  std::map<std::string, std::string> inventory = get_node_inventory();

  // TODO hardcoded
  const std::string ex_path = "data/examples/";
  const QSize       size = QSize(512, 512);

  auto *app = static_cast<hesiod::HesiodApplication *>(QCoreApplication::instance());

  AppContext &ctx = app->get_context();
  ctx.save_state();

  // headless: skip GUI-only / OpenGL widgets (3D viewer) when building the graph UI
  ctx.headless = true;

  // strip everything but the node graph from the snapshot: deactivate the node
  // settings pan, the per-graph 3D viewer, the node library panel (so the graph
  // gets the full frame width), and the WebEngine texture downloader.
  ctx.app_settings.node_editor.show_node_settings_pan = false;
  ctx.app_settings.node_editor.show_viewer = false;
  ctx.app_settings.node_editor.show_node_library_pan = false;
  ctx.app_settings.interface.enable_texture_downloader = false;

  for (auto &[node_type, _] : inventory)
  {
    const std::string fname = ex_path + node_type + ".hsd";

    if (std::filesystem::exists(fname))
    {
      Logger::log()->trace("- default file exists: {}", fname);

      // a single broken example (e.g. a stale port name in the .hsd) must not
      // abort the whole batch: log and skip it
      try
      {
        app->load_project_model_and_ui(fname);

        GraphTabsWidget *p_gtw = app->get_project_ui_ref()->get_graph_tabs_widget_ref();

        if (p_gtw)
        {
          QWidget *widget = dynamic_cast<QWidget *>(p_gtw);

          // size the widget to the final render size FIRST, then fit the graph
          // to it: zoom_to_content() must run against the actual 512x512
          // viewport, otherwise the fit is computed at the pre-render size and
          // the graph is cropped (the old "TODO refit" problem).
          widget->setFixedSize(size);
          QCoreApplication::processEvents();

          // the widget tree is never shown in headless mode, so the resize
          // stays pending and child layouts (incl. the graph view viewport)
          // keep their stale pre-load geometry until the first render. grab()
          // flushes pending resize/layout the same way render() does, so the
          // viewport has its final size before the fit is computed.
          widget->grab();

          p_gtw->zoom_to_content();
          QCoreApplication::processEvents();

          render_widget_screenshot(widget, node_type + "_hsd_example.png", size);

          QCoreApplication::processEvents();

          // to avoid Qt panicking...
          QEventLoop loop;
          QTimer::singleShot(50, &loop, &QEventLoop::quit);
          loop.exec();
        }
      }
      catch (const std::exception &e)
      {
        Logger::log()->error("snapshot generation failed for '{}', skipping: {}",
                             fname,
                             e.what());
      }
    }
  }

  ctx.restore_state();
}

} // namespace hesiod::cli
