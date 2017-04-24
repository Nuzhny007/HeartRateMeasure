#pragma once

#include <cmath>
#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#include <Eigen/Dense>

#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"

// --------------------------------------------------------
// 
// --------------------------------------------------------
class FastICA
{
public:
	FastICA();
	~FastICA();
	// Главный метод, раскладывающий сигнал по компонентам
    void apply(cv::Mat& src, cv::Mat& dst, cv::Mat& W);

private:
	// Матрица масштабных коэффициентов
    Eigen::MatrixXd m_mixing_matrix;
	// Верхнее значение предела по кол-ву итераций алгоритма.
	int max_iter;
	// Малое значение, для задания допустимой погрешности алгоритма.
	double tol;

    static const double alpha; // alpha must be in range [1.0 - 2.0]

    static Eigen::MatrixXd sym_decorrelation(Eigen::MatrixXd mixing_matrix);
    static double gx(double x);
    static double g_x(double x);
	// Главный метод, раскладывающий сигнал по компонентам
    Eigen::MatrixXd apply(Eigen::MatrixXd& features);
}; 

