cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)

project(
  FaceLandmarks
  VERSION 1.0
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/build/conan")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/build/conan")

find_package(OpenCV 4.6.0 REQUIRED)
find_package(tensorflowlite REQUIRED)

add_library(face_lib FaceLib/face_detector.cpp FaceLib/landmark_extractor.cpp)

target_link_libraries(face_lib PUBLIC tensorflow::tensorflowlite opencv_highgui opencv_imgproc)

add_executable(demo demo.cpp)

target_link_libraries(demo PUBLIC face_lib)