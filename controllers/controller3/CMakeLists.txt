cmake_minimum_required(VERSION 3.0)
project(PRM)

set(OpenCV_DIR "C:/Users/xu/Desktop/2021c/OpenCV-MinGW-Build-OpenCV-3.4.8-x64") # 根据自己路径修改
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif()

# include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(C:/Program\ Files/Webots/lib/controller) # 根据自己路径修改
 
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    C:/Program\ Files/Webots/include/controller/c # 根据自己路径修改
    C:/Program\ Files/Webots/include/controller/cpp)

set (LIBRARIES
#Controller.dll
#CppController.dll
${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX}
${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX}
${OpenCV_LIBS}
)
message(STATUS "LIBRARIES = ${LIBRARIES}")

# add_library(${PROJECT_NAME} Bspline.cpp)

# Setup the target executable.
add_executable(controller3 controller3.cpp )
target_link_libraries(controller3 ${LIBRARIES})
