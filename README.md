# VitaGraph

Video pulse detection technology

[![Pulse extraction:](https://img.youtube.com/vi/8ZOPl2qWZD8/0.jpg)](https://youtu.be/8ZOPl2qWZD8)

License: MIT

#### Thirdparty libraries
* [OpenCV](https://github.com/opencv/opencv) and [contrib](https://github.com/opencv/opencv_contrib)
* [OpenVINO](https://github.com/opencv/dldt)
* [Sources by Smorodov](http://www.compvision.ru/forum/index.php?/topic/1512-%D0%B8%D0%B7%D0%BC%D0%B5%D1%80%D0%B8%D1%82%D0%B5%D0%BB%D1%8C-%D0%BF%D1%83%D0%BB%D1%8C%D1%81%D0%B0-%D0%BF%D0%BE-%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8E-%D1%83%D1%87%D0%B0%D1%81%D1%82%D0%BA%D0%B0-%D0%BA%D0%BE%D0%B6%D0%B8/)
* [Eigen 3](https://eigen.tuxfamily.org/dox/)
* Third-party iir
* [vpglib](https://github.com/pi-null-mezon/vpglib)

#### Build

1. Install CMake
2. Install OpenCV (https://github.com/opencv/opencv) and OpenCV contrib (https://github.com/opencv/opencv_contrib) repositories
3. Install Eigen 3 (https://eigen.tuxfamily.org/dox/)
4. Install Qt5
5. Download project sources:
      
        git clone https://github.com/Nuzhny007/HeartRateMeasure.git
        cd HeartRateMeasure
        mkdir build
        cd build
        cmake . .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<opencv_build_dir>
        make

#### Build on Windows

**1. Qt:**
     
    - https://download.qt.io/archive/qt/5.12/5.12.0/ or https://www.qt.io/download
    - Install


**2. boost:**

    - https://www.boost.org/users/download/
    - Build and install: b2.exe -j8 toolset=msvc address-model=64 architecture=x86 link=static threading=multi runtime-link=static,shared --build-type=complete

**3. CMake:**

    - https://cmake.org/download/
    - Install

**4. [optional] Intel OpenVINO:**

    - https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-windows
    - Install

**5. OpenCV:**
   
    - git glone https://github.com/opencv/opencv
    - git glone https://github.com/opencv/opencv_contrib
    - Manuals: https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html or https://www.learnopencv.com/install-opencv3-on-windows/
      In CMake you can disable: BUILD_TESTS and BUILD_PERF_TESTS, BUILD_EXAMPLES - optional.
      You need set OPENCV_EXTRA_MODULES_PATH (for example): C:/work/libraries/opencv/opencv_contrib/modules
      You need set WITH_QT and paths for Qt modules. 
      If OpenVINO was downloaded than set INFERENCE_ENGINE_DIR (for example): C:\Intel\computer_vision_sdk_2018.3.343/deployment_tools/inference_engine/share
    - In CMake gui press configure, generate and Open Project buttons
    - Build Debug and Release versions of OpenCV

**6. Eigen:**

    - Download: http://eigen.tuxfamily.org/index.php?title=Main_Page
    - Extract

**7. Vitagraph:**

    - git clone https://nuzhny007@bitbucket.org/nuzhny007/beatmagnifier.git
    - Use CMake gui for build (checkbox Advanced will be enabled)
      Set Eigen3_DIR (for example): C:/Program Files/Eigen3/share/eigen3/cmake
      Set OpenCV_DIR (it must be build directory with MSVS sln file)
      Set boost dirs, for example Boost_INCLUDE_DIR=C:/work/libraries/boost_1_67_0
      Set InferenceEngine_DIR
    - In CMake gui press configure, generate and Open Project buttons
    - Build Debug and Release versions of program
 

#### Run

1. VitaGraph.exe uses Qt-based GUI
2. HeartRateMeasure.exe is a command line version
Both programs use configuration files, for example pca_128.conf in folder data

#### Citation

```bibtex
@article{ПРОГРАММА ВЫЯВЛЕНИЯ И ВЫЧИСЛЕНИЯ ПУЛЬСА ПО ВИДЕОИЗОБРАЖЕНИЮ,
  title={ПРОГРАММА ВЫЯВЛЕНИЯ И ВЫЧИСЛЕНИЯ ПУЛЬСА ПО ВИДЕОИЗОБРАЖЕНИЮ},
  author={Шарикова, МИ and Нужный, СП},
  journal={свидетельство о государственной регистрации программы для ЭВМ},
  number={RU 2018618445},
  year={2018},
  publisher={Патентное ведомство: Россия}
}
```
