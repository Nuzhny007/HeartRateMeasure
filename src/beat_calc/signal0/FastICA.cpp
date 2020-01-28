#include "FastICA.h"

const double FastICA::alpha = 1.0;
// --------------------------------------------------------
// 
// --------------------------------------------------------
Eigen::MatrixXd FastICA::sym_decorrelation(Eigen::MatrixXd mixing_matrix)
{
    Eigen::MatrixXd K = mixing_matrix * mixing_matrix.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig;
	eig.compute(K);
    return ((eig.eigenvectors() * eig.eigenvalues().cwiseSqrt().asDiagonal().inverse()) * eig.eigenvectors().transpose()) * mixing_matrix;
}
// --------------------------------------------------------
// 
// --------------------------------------------------------

double FastICA::gx(double x)
{
	return tanh(x*alpha);
}
// --------------------------------------------------------
// 
// --------------------------------------------------------
double FastICA::g_x(double x)
{
	return alpha * (1.0 - gx(x)*gx(x));
}
// --------------------------------------------------------
// 
// --------------------------------------------------------
FastICA::FastICA() 
{
	max_iter = 200;
	tol = 1e-10;
}
// --------------------------------------------------------
// 
// --------------------------------------------------------
FastICA::~FastICA()
{

}
// --------------------------------------------------------
// 
// --------------------------------------------------------
void FastICA::apply(cv::Mat& src, cv::Mat& dst, cv::Mat& W)
{
    Eigen::MatrixXd X = Eigen::MatrixXd(src.rows, src.cols);

	cv2eigen(src, X);
	apply(X);
	eigen2cv(X,dst);
	eigen2cv(m_mixing_matrix,W);
}
// --------------------------------------------------------
// 
// --------------------------------------------------------
Eigen::MatrixXd FastICA::apply(Eigen::MatrixXd& X)
{
	int n = X.rows();
	int p = X.cols();
	int m = n;

	// Whiten
    Eigen::VectorXd mean = (X.rowwise().sum() / (double)p);
    Eigen::MatrixXd SPX = X.colwise() - mean;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(SPX, Eigen::ComputeThinU);

    Eigen::MatrixXd u = svd.matrixU();
    Eigen::MatrixXd d = svd.singularValues();

	// see Hyvarinen (6.33) p.140
    Eigen::MatrixXd K = u.transpose();
	for (int r = 0; r < K.rows(); r++)
	{
		K.row(r) /= d(r);
	}
	// see Hyvarinen (13.6) p.267 Here WX is white and data
	// in X has been projected onto a subspace by PCA
    Eigen::MatrixXd WX = K * SPX;
	WX *= sqrt((double)p);

	cv::RNG rng;
	// Initial mixing matrix estimate
	if (m_mixing_matrix.rows() != m || m_mixing_matrix.cols() != m)
	{
        m_mixing_matrix = Eigen::MatrixXd(m,m);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < m; j++)
			{
				m_mixing_matrix(i,j) = rng.gaussian(1);
			}
		}
	}

	m_mixing_matrix = sym_decorrelation(m_mixing_matrix);

	int iter = 0;
	double lim = tol+1;
	while (lim > tol && iter < max_iter)
	{
        Eigen::MatrixXd wtx = m_mixing_matrix * WX;
        Eigen::MatrixXd gwtx  = wtx.unaryExpr(std::cref(gx));
        Eigen::MatrixXd g_wtx = wtx.unaryExpr(std::cref(g_x));
        Eigen::MatrixXd W1 = (gwtx * WX.transpose()) / (double)p - (g_wtx.rowwise().sum()/(double)p).asDiagonal() * m_mixing_matrix;
		W1 = sym_decorrelation(W1);
		lim = ((W1 * m_mixing_matrix.transpose()).diagonal().cwiseAbs().array() - 1.0).abs().maxCoeff();
		m_mixing_matrix = W1;
		iter++;
	}

	// Unmix
	m_mixing_matrix = (m_mixing_matrix*K);
	X = m_mixing_matrix * X;
	// set m_mixing_matrix
	m_mixing_matrix = m_mixing_matrix.inverse();
	return X;
}
