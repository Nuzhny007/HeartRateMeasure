# HeartRateMeasure

Video based Heart rate measure: [main project](http://people.csail.mit.edu/mrub/vidmag/) and [paper](http://people.csail.mit.edu/mrub/vidmag/papers/Balakrishnan_Detecting_Pulse_from_2013_CVPR_paper.pdf)

License: GNU GPLv3 http://www.gnu.org/licenses/gpl-3.0.txt 

In project uses libraries:
- [OpenCV](https://github.com/opencv/opencv) and [contrib](https://github.com/opencv/opencv_contrib)
- [Sources by Smorodov](http://www.compvision.ru/forum/index.php?/topic/1512-%D0%B8%D0%B7%D0%BC%D0%B5%D1%80%D0%B8%D1%82%D0%B5%D0%BB%D1%8C-%D0%BF%D1%83%D0%BB%D1%8C%D1%81%D0%B0-%D0%BF%D0%BE-%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8E-%D1%83%D1%87%D0%B0%D1%81%D1%82%D0%BA%D0%B0-%D0%BA%D0%BE%D0%B6%D0%B8/)
- [Eigen 3](https://eigen.tuxfamily.org/dox/)
- Third-party iir

#### Build

1. Install CMake
2. Install OpenCV (https://github.com/opencv/opencv) and OpenCV contrib (https://github.com/opencv/opencv_contrib) repositories
3. Install Eigen 3 (https://eigen.tuxfamily.org/dox/)
4. Download project sources:
      
        git clone https://github.com/Nuzhny007/HeartRateMeasure.git
        cd HeartRateMeasure
        mkdir build
        cd build
        cmake . .. -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=<opencv_build_dir>
        make

