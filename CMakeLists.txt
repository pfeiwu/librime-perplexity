project(rime-perplexity)
cmake_minimum_required(VERSION 3.10)

aux_source_directory(src perplexity_src)

add_library(rime-perplexity-objs OBJECT ${perplexity_src})
target_include_directories(rime-perplexity-objs
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")

set(PERPLEXITY_LLAMA_CPP_DIR ""
  CACHE PATH "Path to llama.cpp source tree for optional perplexity scoring")
set(PERPLEXITY_LLAMA_CPP_BUILD_DIR ""
  CACHE PATH "Path to llama.cpp build tree for optional perplexity scoring")
set(PERPLEXITY_ONNXRUNTIME_DIR ""
  CACHE PATH "Path to onnxruntime install root for optional masked LM scoring")

# Auto-discover thirdparty/ layout populated by action-install.sh.
# This is the standard path: bash install-plugins.sh run=plugins/perplexity/action-install.sh ...
set(perplexity_thirdparty_dir "${CMAKE_CURRENT_LIST_DIR}/thirdparty")
if(NOT PERPLEXITY_LLAMA_CPP_DIR AND
   EXISTS "${perplexity_thirdparty_dir}/llama.cpp/include/llama.h")
  set(PERPLEXITY_LLAMA_CPP_DIR "${perplexity_thirdparty_dir}/llama.cpp")
  message(STATUS
    "perplexity: auto-discovered llama.cpp at ${PERPLEXITY_LLAMA_CPP_DIR}")
endif()
if(NOT PERPLEXITY_LLAMA_CPP_BUILD_DIR AND PERPLEXITY_LLAMA_CPP_DIR AND
   IS_DIRECTORY "${PERPLEXITY_LLAMA_CPP_DIR}/build/bin")
  set(PERPLEXITY_LLAMA_CPP_BUILD_DIR "${PERPLEXITY_LLAMA_CPP_DIR}/build")
endif()
if(NOT PERPLEXITY_ONNXRUNTIME_DIR AND
   EXISTS "${perplexity_thirdparty_dir}/onnxruntime/include/onnxruntime_cxx_api.h")
  set(PERPLEXITY_ONNXRUNTIME_DIR "${perplexity_thirdparty_dir}/onnxruntime")
  message(STATUS
    "perplexity: auto-discovered onnxruntime at ${PERPLEXITY_ONNXRUNTIME_DIR}")
endif()

function(perplexity_collect_runtime_libs output_var lib_dir)
  set(collected)
  foreach(lib_name ${ARGN})
    file(GLOB lib_files
      "${lib_dir}/lib${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX}"
      "${lib_dir}/lib${lib_name}${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
    list(APPEND collected ${lib_files})
  endforeach()
  if(collected)
    list(REMOVE_DUPLICATES collected)
  endif()
  set(${output_var} ${collected} PARENT_SCOPE)
endfunction()

set(perplexity_runtime_deps)
set(perplexity_llama_deps)
set(perplexity_llama_lib_dir "${PERPLEXITY_LLAMA_CPP_BUILD_DIR}/bin")
set(perplexity_shared_suffix "${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(EXISTS "${PERPLEXITY_LLAMA_CPP_DIR}/include/llama.h" AND
   EXISTS "${perplexity_llama_lib_dir}/libllama${perplexity_shared_suffix}")
  target_compile_definitions(rime-perplexity-objs
    PRIVATE RIME_PERPLEXITY_ENABLE_LLAMA)
  target_include_directories(rime-perplexity-objs
    PRIVATE
      "${PERPLEXITY_LLAMA_CPP_DIR}/include"
      "${PERPLEXITY_LLAMA_CPP_DIR}/ggml/include")
  foreach(ggml_lib
          llama
          ggml
          ggml-base
          ggml-cpu
          ggml-metal
          ggml-blas
          ggml-vulkan
          ggml-cuda
          ggml-rpc)
    if(EXISTS "${perplexity_llama_lib_dir}/lib${ggml_lib}${perplexity_shared_suffix}")
      list(APPEND perplexity_llama_deps
        "${perplexity_llama_lib_dir}/lib${ggml_lib}${perplexity_shared_suffix}")
    endif()
  endforeach()
  perplexity_collect_runtime_libs(perplexity_llama_runtime_deps
    "${perplexity_llama_lib_dir}"
    llama
    ggml
    ggml-base
    ggml-cpu
    ggml-metal
    ggml-blas
    ggml-vulkan
    ggml-cuda
    ggml-rpc)
  list(APPEND perplexity_runtime_deps ${perplexity_llama_runtime_deps})
  message(STATUS "perplexity: enabling llama.cpp causal LM scorer")
else()
  message(STATUS "perplexity: llama.cpp not found, causal LM scorer disabled")
endif()

set(perplexity_onnx_deps)
set(perplexity_onnx_lib_dir "${PERPLEXITY_ONNXRUNTIME_DIR}/lib")
if(PERPLEXITY_ONNXRUNTIME_DIR AND
   EXISTS "${PERPLEXITY_ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" AND
   EXISTS "${perplexity_onnx_lib_dir}/libonnxruntime${perplexity_shared_suffix}")
  target_compile_definitions(rime-perplexity-objs
    PRIVATE RIME_PERPLEXITY_ENABLE_ONNX)
  target_include_directories(rime-perplexity-objs
    PRIVATE "${PERPLEXITY_ONNXRUNTIME_DIR}/include")
  if(EXISTS "${perplexity_onnx_lib_dir}/libonnxruntime_providers_cuda${perplexity_shared_suffix}")
    target_compile_definitions(rime-perplexity-objs
      PRIVATE RIME_PERPLEXITY_HAVE_ORT_CUDA)
  endif()
  foreach(onnx_lib
          onnxruntime
          onnxruntime_providers_shared)
    if(EXISTS "${perplexity_onnx_lib_dir}/lib${onnx_lib}${perplexity_shared_suffix}")
      list(APPEND perplexity_onnx_deps
        "${perplexity_onnx_lib_dir}/lib${onnx_lib}${perplexity_shared_suffix}")
    endif()
  endforeach()
  perplexity_collect_runtime_libs(perplexity_onnx_runtime_deps
    "${perplexity_onnx_lib_dir}"
    onnxruntime
    onnxruntime_providers_shared
    onnxruntime_providers_cuda
    onnxruntime_providers_coreml
    onnxruntime_providers_dml)
  list(APPEND perplexity_runtime_deps ${perplexity_onnx_runtime_deps})
  message(STATUS "perplexity: enabling onnxruntime masked LM scorer")
else()
  message(STATUS "perplexity: onnxruntime not found, masked LM scorer disabled")
endif()

if(NOT perplexity_llama_deps AND NOT perplexity_onnx_deps)
  message(FATAL_ERROR
    "perplexity: at least one scorer backend must be enabled.\n"
    "  Pick one of:\n"
    "    1. Recommended — fetch & build dependencies into plugins/perplexity/thirdparty:\n"
    "         bash install-plugins.sh run=plugins/perplexity/action-install.sh pfeiwu/librime-perplexity\n"
    "       (set PERPLEXITY_LLAMA_BACKEND=cuda|vulkan|metal|hip for GPU; PERPLEXITY_ORT_BACKEND=cuda for ORT GPU)\n"
    "    2. Point at an existing llama.cpp build:\n"
    "         -DPERPLEXITY_LLAMA_CPP_DIR=<src> -DPERPLEXITY_LLAMA_CPP_BUILD_DIR=<build>\n"
    "    3. Point at an existing onnxruntime install:\n"
    "         -DPERPLEXITY_ONNXRUNTIME_DIR=<root>")
endif()

if(BUILD_SHARED_LIBS)
  set_target_properties(rime-perplexity-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(perplexity-tokenizer-probe EXCLUDE_FROM_ALL
  tools/tokenizer_probe.cc
  src/bert_wordpiece_tokenizer.cc)
target_include_directories(perplexity-tokenizer-probe
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
target_link_libraries(perplexity-tokenizer-probe ${rime_library})

if(perplexity_llama_deps)
  add_executable(perplexity-causal-score-probe EXCLUDE_FROM_ALL
    tools/causal_score_probe.cc)
  target_include_directories(perplexity-causal-score-probe
    PRIVATE
      "${PERPLEXITY_LLAMA_CPP_DIR}/include"
      "${PERPLEXITY_LLAMA_CPP_DIR}/ggml/include")
  target_link_libraries(perplexity-causal-score-probe
    ${perplexity_llama_deps})
endif()

if(perplexity_onnx_deps)
  add_executable(perplexity-masked-score-probe EXCLUDE_FROM_ALL
    tools/masked_score_probe.cc
    src/onnx_masked_scorer.cc
    src/bert_wordpiece_tokenizer.cc)
  target_compile_definitions(perplexity-masked-score-probe
    PRIVATE RIME_PERPLEXITY_ENABLE_ONNX)
  if(EXISTS "${perplexity_onnx_lib_dir}/libonnxruntime_providers_cuda${perplexity_shared_suffix}")
    target_compile_definitions(perplexity-masked-score-probe
      PRIVATE RIME_PERPLEXITY_HAVE_ORT_CUDA)
  endif()
  target_include_directories(perplexity-masked-score-probe
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/src"
      "${PERPLEXITY_ONNXRUNTIME_DIR}/include")
  target_link_libraries(perplexity-masked-score-probe
    ${rime_library}
    ${perplexity_onnx_deps})
endif()

set(plugin_name rime-perplexity PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-perplexity-objs> PARENT_SCOPE)
set(plugin_deps
  ${rime_library}
  ${rime_gears_library}
  ${perplexity_llama_deps}
  ${perplexity_onnx_deps}
  PARENT_SCOPE)
set(plugin_modules "perplexity" PARENT_SCOPE)

if(NOT BUILD_MERGED_PLUGINS AND perplexity_runtime_deps)
  list(REMOVE_DUPLICATES perplexity_runtime_deps)
  set(perplexity_runtime_install_files)
  foreach(runtime_dep IN LISTS perplexity_runtime_deps)
    string(APPEND perplexity_runtime_install_files
      "\n      [=[${runtime_dep}]=]")
  endforeach()
  install(CODE "
    file(INSTALL
      DESTINATION \"${CMAKE_INSTALL_FULL_LIBDIR}/${RIME_PLUGINS_DIR}/perplexity\"
      TYPE FILE
      FOLLOW_SYMLINK_CHAIN
      FILES ${perplexity_runtime_install_files})
  ")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
    if(APPLE)
      set(perplexity_install_rpath "@loader_path/perplexity")
    else()
      set(perplexity_install_rpath "$ORIGIN/perplexity")
    endif()
    get_filename_component(perplexity_plugins_binary_dir
      "${CMAKE_CURRENT_BINARY_DIR}" DIRECTORY)
    cmake_language(DEFER DIRECTORY "${perplexity_plugins_binary_dir}"
      CALL set_target_properties rime-perplexity
      PROPERTIES INSTALL_RPATH "${perplexity_install_rpath}")
  else()
    message(WARNING
      "perplexity: CMake 3.19 or newer is required to set install RPATH "
      "for bundled runtime libraries")
  endif()
endif()
