project(rime-perplexity)
cmake_minimum_required(VERSION 3.10)

aux_source_directory(src perplexity_src)

add_library(rime-perplexity-objs OBJECT ${perplexity_src})

set(PERPLEXITY_LLAMA_CPP_DIR ""
  CACHE PATH "Path to llama.cpp source tree for optional perplexity scoring")
set(PERPLEXITY_LLAMA_CPP_BUILD_DIR ""
  CACHE PATH "Path to llama.cpp build tree for optional perplexity scoring")

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
  message(FATAL_ERROR
    "perplexity: llama.cpp headers or shared libraries not found. "
    "Set PERPLEXITY_LLAMA_CPP_DIR and PERPLEXITY_LLAMA_CPP_BUILD_DIR.")
endif()

if(BUILD_SHARED_LIBS)
  set_target_properties(rime-perplexity-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

set(plugin_name rime-perplexity PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-perplexity-objs> PARENT_SCOPE)
set(plugin_deps
  ${rime_library}
  ${rime_gears_library}
  ${perplexity_llama_deps}
  PARENT_SCOPE)
set(plugin_modules "perplexity" PARENT_SCOPE)
