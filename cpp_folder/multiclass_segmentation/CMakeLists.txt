cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(multiclass_segmentation)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package( Boost REQUIRED COMPONENTS program_options regex )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( ${Boost_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_executable(multiclass_segmentation multiclass_segmentation.cpp)

target_link_libraries(multiclass_segmentation "${TORCH_LIBRARIES}")
target_link_libraries(multiclass_segmentation ${OpenCV_LIBS})
target_link_libraries(multiclass_segmentation ${Boost_REGEX_LIBRARY} )

set_property(TARGET multiclass_segmentation PROPERTY CXX_STANDARD 17)




# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET multiclass_segmentation
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:multiclass_segmentation>)
endif (MSVC)
