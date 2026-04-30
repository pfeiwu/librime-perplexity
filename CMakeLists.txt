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
  message(STATUS "perplexity: enabling onnxruntime masked LM scorer")
else()
  message(STATUS "perplexity: onnxruntime not found, masked LM scorer disabled")
endif()

if(NOT perplexity_llama_deps AND NOT perplexity_onnx_deps)
  message(FATAL_ERROR
    "perplexity: at least one scorer backend must be enabled. "
    "Set PERPLEXITY_LLAMA_CPP_DIR/PERPLEXITY_LLAMA_CPP_BUILD_DIR "
    "(causal) or PERPLEXITY_ONNXRUNTIME_DIR (masked).")
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
