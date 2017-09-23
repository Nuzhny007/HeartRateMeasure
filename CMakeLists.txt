cmake_minimum_required (VERSION 2.8)

project (HeartRateMeasure)

FIND_PACKAGE(OpenCV 3.0 REQUIRED)

FIND_PACKAGE(Eigen)

unset(CMAKE_CXX_FLAGS CACHE)

find_package(OpenMP)
if (OPENMP_FOUND)
    list(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    list(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic-errors --std=c++0x" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native --fast-math -ffast-math -funroll-loops -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS" CACHE STRING COMPILE_FLAGS FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -march=native -mtune=native -Wall -DDEBUG" CACHE STRING COMPILE_FLAGS FORCE)
elseif (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W4" CACHE STRING COMPILE_FLAGS FORCE)
endif()

option(USE_GPU "Should the GPU be used?" OFF)
if(USE_GPU)
    add_definitions(-DUSE_GPU)
endif(USE_GPU)


include_directories(
${OpenCV_INCLUDE_DIRS}
"/usr/include/eigen3"
)

# ----------------------------------------------------------------------
set(SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/FaceDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/SkinDetector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/LKTracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/EulerianMA.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/iir.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/stat.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SignalProcessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FastICA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pca.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/FaceDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/SkinDetector.h
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_track/LKTracker.h
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/EulerianMA.h
    ${CMAKE_CURRENT_SOURCE_DIR}/eulerian_ma/iir.h
)

set(LIBS
    ${OpenCV_LIBS}
)

add_executable(HeartRateMeasure ${SOURCE} ${HEADERS})

target_link_libraries(HeartRateMeasure ${LIBS})
