#pragma once
#include <opencv2/opencv.hpp>

bool MakePCA(const std::deque<cv::Mat>& images, cv::Mat& resImg);
bool MakePCA(const cv::Mat& src, cv::Mat& dst);
