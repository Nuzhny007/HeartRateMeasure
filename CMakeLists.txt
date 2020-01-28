cmake_minimum_required(VERSION 3.5)

project(BeatMagnifier)

FIND_PACKAGE(OpenCV 4.0 REQUIRED)

FIND_PACKAGE(Eigen3)

set(Boost_USE_STATIC_LIBS    ON)
set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost COMPONENTS system filesystem program_options REQUIRED)

find_package(InferenceEngine 1.2)

unset(CMAKE_CXX_FLAGS CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE})
elseif(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors --std=c++14" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
    add_definitions(-DUSE_OPENCV)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX /fp:fast /O2 /Ob2 /Oi /Ot" CACHE STRING COMPILE_FLAGS FORCE)
    add_definitions(-DWIN32 -D_USE_MATH_DEFINES -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -DUSE_OPENCV -DUSE_STATIC_IE)
endif()

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${InferenceEngine_INCLUDE_DIRS})

if (EIGEN3_FOUND)
  INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")
else()
if (CMAKE_COMPILER_IS_GNUCXX)
  INCLUDE_DIRECTORIES("/usr/include/eigen3")
elseif (MSVC)
  INCLUDE_DIRECTORIES("c:/work/libraries/eigen3")
endif()
endif()
# ----------------------------------------------------------------------

add_subdirectory(src)
add_subdirectory(gui)
add_subdirectory(test)

# ----------------------------------------------------------------------

set(DATA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/data/skin_model.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/data/pca_128.conf
    ${CMAKE_CURRENT_SOURCE_DIR}/data/haarcascades/haarcascade_frontalface_alt2.xml
    ${CMAKE_CURRENT_SOURCE_DIR}/data/face_detector/deploy.prototxt
    ${CMAKE_CURRENT_SOURCE_DIR}/data/face_detector/face_landmark_model.dat
    ${CMAKE_CURRENT_SOURCE_DIR}/data/face_detector/res10_300x300_ssd_iter_140000.caffemodel
)

if (CMAKE_COMPILER_IS_GNUCXX)
    file(COPY ${DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data)
elseif(MSVC)
    file(COPY ${DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/data)
    file(COPY ${DATA_FILES} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/data)
endif()
