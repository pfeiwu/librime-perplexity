#!/usr/bin/env bash
# Fetch and prepare third-party dependencies for the perplexity plugin.
#
# Invoked by librime's install-plugins.sh:
#   bash install-plugins.sh run=plugins/perplexity/action-install.sh pfeiwu/librime-perplexity
#
# Layout produced (consumed by plugins/perplexity/CMakeLists.txt auto-discovery):
#   plugins/perplexity/thirdparty/llama.cpp/        # cloned + built
#   plugins/perplexity/thirdparty/onnxruntime/      # extracted prebuilt tarball
#
# Knobs (env vars):
#   PERPLEXITY_LLAMA_BACKEND   cpu|metal|cuda|vulkan|hip   (default: metal on macOS, cpu elsewhere)
#   PERPLEXITY_ORT_BACKEND     cpu|cuda                    (default: cpu)
#   PERPLEXITY_LLAMA_REPO      git URL                     (default: upstream ggerganov/llama.cpp)
#   PERPLEXITY_LLAMA_REF       branch / tag                (default: master)
#   PERPLEXITY_ORT_VERSION     onnxruntime release version (default: 1.20.1)

set -euo pipefail

plugin="${1:-}"
plugin_dir="${2:-plugins/perplexity}"

if [[ ! -d "$plugin_dir" ]]; then
  echo "[perplexity] plugin directory not found: $plugin_dir" >&2
  exit 1
fi

thirdparty="$plugin_dir/thirdparty"
mkdir -p "$thirdparty"

os="$(uname -s)"
arch="$(uname -m)"

if [[ -z "${PERPLEXITY_LLAMA_BACKEND:-}" ]]; then
  case "$os" in
    Darwin) PERPLEXITY_LLAMA_BACKEND=metal ;;
    *)      PERPLEXITY_LLAMA_BACKEND=cpu   ;;
  esac
fi
PERPLEXITY_ORT_BACKEND="${PERPLEXITY_ORT_BACKEND:-cpu}"

llama_repo="${PERPLEXITY_LLAMA_REPO:-https://github.com/ggerganov/llama.cpp.git}"
llama_ref="${PERPLEXITY_LLAMA_REF:-master}"
ort_version="${PERPLEXITY_ORT_VERSION:-1.20.1}"

# -----------------------------------------------------------------------------
# llama.cpp
# -----------------------------------------------------------------------------
llama_dir="$thirdparty/llama.cpp"
if [[ ! -d "$llama_dir/.git" ]]; then
  echo "[perplexity] cloning llama.cpp ($llama_ref) into $llama_dir"
  git clone --depth=1 --branch "$llama_ref" "$llama_repo" "$llama_dir"
else
  echo "[perplexity] llama.cpp already present at $llama_dir, reusing"
fi

llama_flags=(
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DLLAMA_BUILD_TOOLS=OFF
  -DLLAMA_CURL=OFF
)
case "$PERPLEXITY_LLAMA_BACKEND" in
  cpu)
    llama_flags+=(-DGGML_METAL=OFF)
    ;;
  metal)
    llama_flags+=(-DGGML_METAL=ON)
    ;;
  cuda)
    llama_flags+=(-DGGML_CUDA=ON)
    ;;
  vulkan)
    llama_flags+=(-DGGML_VULKAN=ON)
    ;;
  hip|rocm)
    llama_flags+=(-DGGML_HIP=ON)
    ;;
  *)
    echo "[perplexity] unknown PERPLEXITY_LLAMA_BACKEND='$PERPLEXITY_LLAMA_BACKEND'" >&2
    echo "             expected one of: cpu metal cuda vulkan hip" >&2
    exit 1
    ;;
esac

echo "[perplexity] configuring llama.cpp (backend=$PERPLEXITY_LLAMA_BACKEND)"
cmake -S "$llama_dir" -B "$llama_dir/build" "${llama_flags[@]}"
echo "[perplexity] building llama.cpp"
cmake --build "$llama_dir/build" --config Release

# -----------------------------------------------------------------------------
# onnxruntime (prebuilt tarball)
# -----------------------------------------------------------------------------
ort_link="$thirdparty/onnxruntime"
if [[ -e "$ort_link/include/onnxruntime_cxx_api.h" ]]; then
  echo "[perplexity] onnxruntime already present at $ort_link, reusing"
else
  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64)
          if [[ "$PERPLEXITY_ORT_BACKEND" == "cuda" ]]; then
            ort_tarball="onnxruntime-linux-x64-gpu-${ort_version}.tgz"
          else
            ort_tarball="onnxruntime-linux-x64-${ort_version}.tgz"
          fi
          ;;
        aarch64|arm64)
          ort_tarball="onnxruntime-linux-aarch64-${ort_version}.tgz"
          ;;
        *)
          echo "[perplexity] unsupported Linux architecture: $arch" >&2
          exit 1
          ;;
      esac
      ;;
    Darwin)
      case "$arch" in
        arm64|aarch64) ort_tarball="onnxruntime-osx-arm64-${ort_version}.tgz" ;;
        x86_64|amd64)  ort_tarball="onnxruntime-osx-x86_64-${ort_version}.tgz" ;;
        *)
          echo "[perplexity] unsupported macOS architecture: $arch" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "[perplexity] unsupported OS: $os (set PERPLEXITY_ONNXRUNTIME_DIR manually)" >&2
      exit 1
      ;;
  esac

  ort_extract="${ort_tarball%.tgz}"
  ort_url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_tarball}"

  echo "[perplexity] downloading $ort_url"
  rm -rf "$thirdparty/$ort_extract"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$ort_url" | tar xz -C "$thirdparty"
  elif command -v wget >/dev/null 2>&1; then
    wget -O - "$ort_url" | tar xz -C "$thirdparty"
  else
    echo "[perplexity] need curl or wget to download onnxruntime" >&2
    exit 1
  fi

  ln -sfn "$ort_extract" "$ort_link"
fi

echo
echo "[perplexity] dependencies ready under $thirdparty"
echo "  llama.cpp    : $llama_dir/build  (backend=$PERPLEXITY_LLAMA_BACKEND)"
echo "  onnxruntime  : $ort_link         (backend=$PERPLEXITY_ORT_BACKEND)"
echo
echo "Next: run 'make' from the librime root to build the plugin."
