# PR #758 Linux investigation

PR: https://github.com/ottolink-dev/Hesiod/pull/758
Tested head: `3ad443e3903fac2f841a0252d12a57ba111575f6`.
Environment: Linux x86_64, Qt 6.11.2, GCC 15.3.0; Xvfb at 1920x1080,
software OpenGL. Separate spot checks used XWayland and native Wayland on
the local desktop. This does not cover Otto's particular Qt version, GPU,
desktop, or mixed-monitor setup.

## Confirmed startup config bug

`ui_scale::executable_dir()` used the running executable's location on Windows,
but resolved `argv[0]` relative to the working directory on Linux. Launching
through PATH or a symlink could therefore read the scale from a different
config than `AppContext` subsequently loads. This is separate from the reported
rendering corruption and can also prevent a saved scale from being applied.

The Linux implementation now resolves `/proc/self/exe`, retaining the existing
fallback when procfs is unavailable. The executable-path tests already present
in the PR failed four checks at every tested scale before this change. An
additional assertion covers resolving the path before QApplication exists.

## Existing suite

Built the PR's `tests/ui_palette` sources with a temporary standalone CMake
wrapper, using Qt Widgets/Test, spdlog, and nlohmann_json. After the path fix,
all 152 checks passed in each of seven separate Xvfb processes with
`QT_SCALE_FACTOR` values 0.5, 0.9, 1, 1.25, 1.5, 2, and 3.

The suite does not construct the real graph editor, node settings, or application
settings content. Passing it alone cannot establish that the reported problem
is fixed. Offscreen mode also does not exercise OpenGL composition.

On the tiling desktop, two XWayland dialog geometry checks failed at 0.9;
they passed on the isolated Xvfb display. One native Wayland flyout-open check
failed. Those results should not be silently counted as passes or confused with
a reproduction of the main-window corruption.

## Rendering diagnosis

The attached report video shows corruption across the main window, not just a
collapsed node settings splitter. A minimal Qt/OpenGL window renders correctly
at 0.9 on the isolated X11 display with this Qt version.

The previous rounding-policy explanation is not established: Qt 6 already uses
PassThrough by default, and QT_SCALE_FACTOR is applied without rounding.
Qt's source also documents limitations for factors below 1 when clamping
screen scale factors. This warrants testing the actual affected platform; it
is not, by itself, evidence that every sub-100% setting fails.

References:

- https://doc.qt.io/qt-6/highdpi.html
- https://github.com/qt/qtbase/blob/6.11/src/gui/kernel/qhighdpiscaling.cpp

## Full application results

The full application built successfully with all pinned submodules unchanged:

```sh
cmake -S /tmp/hesiod-758 -B /tmp/hesiod-758-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DHESIOD_ENABLE_GENERATE_APP_IMAGE=OFF \
  -DHESIOD_ENABLE_PCH=ON -DHIGHMAP_ENABLE_BENCHMARKS=OFF
cmake --build /tmp/hesiod-758-build -j 16
```

A temporary diagnostic executable links the actual application objects with a
small replacement main. It applies the persisted scale before constructing
HesiodApplication, creates a Noise node, clicks its header through X11 using
xdotool, checks selection and populated NodeAttributesWidget content, opens
the actual AppSettingsWindow inside ScrollableDialog, and captures the windows.
It uses an isolated portable config beside the test binary. OpenGL reported
Mesa llvmpipe, OpenGL 4.6 compatibility profile, LLVM 21.1.8.

- At 0.9, 1, 1.25, and 1.5, native node selection and settings population pass,
  and the full application settings dialog fits the screen.
- At 0.9, the node settings pane is also entirely inside the screen. The
  main-window corruption from Otto's video is **not reproduced** here.
- At 2 and 3, native node selection and settings population still pass once
  the node header is scrolled into view, but the node settings pane extends
  beyond the screen. The main window's layout minimum exceeds the available
  logical width. `Viewer::Viewer` sets a minimum size from the viewer settings;
  clamping the main window before layout does not overcome those constraints.
  This is an outstanding high-scale usability issue, not evidence of the 0.9
  corruption. The production layout was not changed in this investigation.

The executable-path fix is the only production behavior change. It does not
justify declaring the reported rendering problem fixed or approving the PR.
Next reproduction needs Otto's Qt version, X11/Wayland backend, GPU/driver,
monitor scale, and preferably the relevant saved window/viewer settings.
Mixed-monitor behavior and full-app native Wayland behavior remain unverified.

To repeat the local diagnostics (requires the downloaded Xvfb and xdotool):

```sh
/tmp/hesiod-758-ui-harness/run-isolated.sh
python3 /tmp/hesiod-758-ui-harness/compile-app-probe.py
/tmp/hesiod-758-ui-harness/run-app.sh
```

The app diagnostic intentionally reports failures for the unresolved 200%/300%
screen overflow. Its logs, rather than the shell script's exit status, contain
the individual results.

## Local artifacts

- Worktree: `/tmp/hesiod-758`, branch `fix/758-linux-ui-scaling`.
- Full build: `/tmp/hesiod-758-build`.
- Diagnostic sources and scripts: `/tmp/hesiod-758-ui-harness`.
- Test logs and images: `/tmp/hesiod-758-results`.
