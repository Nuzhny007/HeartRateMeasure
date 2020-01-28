#include "EulerianMA.h"
#include <math.h>
#include <iomanip>
#include "iir.h"

///
/// \brief rgb2ntsc
/// RGB 2 YIQ conversion, compared with Matlab, it works well
/// \param src
/// \param dst
///
void rgb2ntsc(cv::Mat rgbFrame, cv::Mat ntscFrame)
{
	const float k = 1.f / 255.f;

    for (int y = 0; y < rgbFrame.rows; ++y)
    {
        const uchar* pRGB = rgbFrame.ptr<uchar>(y);
		float* pNTSC = ntscFrame.ptr<float>(y);

        for (int x = 0; x < rgbFrame.cols; ++x)
        {
            float v0 = k * pRGB[0];
            float v1 = k * pRGB[1];
            float v2 = k * pRGB[2];

            pNTSC[2] = (0.299f * v2 + 0.587f * v1 + 0.114f * v0) ;
            pNTSC[1] = (0.595716f * v2 - 0.274453f * v1 - 0.321263f * v0);
            pNTSC[0] = (0.211456f * v2 - 0.522591f * v1 + 0.311135f * v0);

            pRGB += 3;
			pNTSC += 3;
        }
    }
}

///
/// \brief ntsc2rgb
/// YIQ 2 RGB convert
/// \param src
/// \param dst
///
void ntsc2rgb(cv::Mat img, float chromAttenuation, cv::Mat ntscFrame)
{
    for (int y = 0; y < img.rows; ++y)
    {
        float* pimg = img.ptr<float>(y);
		const float* pntsc = ntscFrame.ptr<float>(y);

        for (int x = 0; x < img.cols; ++x)
        {
            float v0 = pimg[0] + pntsc[0];
            float v1 = chromAttenuation * pimg[1] + pntsc[1];
            float v2 = chromAttenuation * pimg[2] + pntsc[2];

			float im0 = 255.f * (1.0f * v2 - 1.107f * v1 + 1.7046f * v0);
            float im1 = 255.f * (1.0f * v2 - 0.2721f * v1 - 0.6474f * v0);
			float im2 = 255.f * (1.0f * v2 + 0.9563f * v1 + 0.621f * v0);

			pimg[0] = std::min(255.f, std::max(0.f, im0));
			pimg[1] = std::min(255.f, std::max(0.f, im1));
			pimg[2] = std::min(255.f, std::max(0.f, im2));
			
            pimg += 3;
			pntsc += 3;
        }
    }
}

///
/// \brief maxPyrHt
/// compute maximum pyramid height of given image and filter sizes.
/// \param imsz
/// \param filtsz
/// \return
///
int maxPyrHt(cv::Size imsz, cv::Size filtsz)
{
    // assume 2D image
    if (imsz.height < filtsz.height || imsz.width < filtsz.width)
    {
        return 0;
    }
    else
    {
        return 1 + maxPyrHt(imsz / 2, filtsz);
    }
}

///
/// \brief lappyr
/// Laplacian pyramid
/// level = 0 means 'auto', i.e. full stack
/// \param src
/// \param level
/// \param lap_arr
/// \param nind
///
void EulerianMA::lappyr(cv::Mat& src, int level, std::vector<cv::Mat>& lap_arr, std::vector<cv::Size>& nind)
{
    lap_arr.clear();
    
    cv::Mat tmp = src.clone();

    int max_ht = maxPyrHt(src.size(), cv::Size(4, 4));

    if (level == 0) // 'auto' full pyr stack
    {
        level = max_ht;
    }

    for (int l = 0; l < level - 1; l++)
    {
        cv::pyrDown(tmp, m_tmpDown);
        cv::pyrUp(m_tmpDown, m_tmpUp, tmp.size());
        m_tmpDst = tmp - m_tmpUp;
        lap_arr.push_back(m_tmpDst);
        nind.push_back(m_tmpDst.size());
        tmp = m_tmpDown;
        swap(tmp, m_tmpDown);
        if (m_tmpDown.rows < 2)
        {
            std::cout << "Laplacian pyramid stop at level" << l << ", due to the image size" <<  std::endl;
        }
    }

    lap_arr.push_back(m_tmpDown);
    nind.push_back(m_tmpDown.size());
}

///
/// \brief reconLpyr
/// Image reconstruction from Laplace Pyramid Vector
/// \param lpyr
/// \param dst
///
void reconLpyr(std::vector<cv::Mat>& lpyr, cv::Mat& dst)
{
    dst = lpyr.back();
    for (int i = static_cast<int>(lpyr.size()) - 2; i >= 0; i--)
    {
        pyrUp(dst, dst, cv::Size(lpyr[i].cols, lpyr[i].rows));
        dst += lpyr[i];
    }
}

///
/// \brief operator *
/// \param f
/// \param foo
/// \return
///
std::vector<cv::Mat> operator* (float f, const std::vector<cv::Mat>& foo)
{
    std::vector<cv::Mat> r;
    for (std::vector<cv::Mat>::const_iterator it = foo.begin(); it != foo.end(); it++)
    {
        r.push_back(*it * f);
    }
    return r;
}

///
/// \brief operator /
/// \param foo
/// \param f
/// \return
///
std::vector<cv::Mat> operator/ (const std::vector<cv::Mat>& foo, float f)
{
    std::vector<cv::Mat> r;
    const float factor = 1.0f / f;
    for (std::vector<cv::Mat>::const_iterator it = foo.begin(); it != foo.end(); it++)
    {
        r.push_back(*it * factor);
    }
    return r;
}

///
/// \brief operator -
/// \param a
/// \param b
/// \return
///
std::vector<cv::Mat> operator- (const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b)
{
    assert(a.size() == b.size());
    std::vector<cv::Mat> r;
    for (std::vector<cv::Mat >::const_iterator ita = a.begin(), itb = b.begin(); ita != a.end(); ita++, itb++)
    {
        r.push_back(*ita - *itb);
    }
    return r;
}

///
/// \brief operator +
/// \param a
/// \param b
/// \return
///
std::vector<cv::Mat> operator+ (const std::vector<cv::Mat>& a, const std::vector<cv::Mat>& b)
{
    assert(a.size() == b.size());
    std::vector<cv::Mat> r;
    for (std::vector<cv::Mat>::const_iterator ita = a.begin(), itb = b.begin(); ita != a.end(); ita++, itb++)
    {
        r.push_back(*ita + *itb);
    }
    return r;
}

///
/// \brief EulerianMA::EulerianMA
///
EulerianMA::EulerianMA()
    :
      m_chromAttenuation(1.0),
      m_alpha(10),
      m_lambda_c(16),
      m_exaggeration_factor(2.0),
      low_b(nullptr),
      high_b(nullptr)
{
    m_delta = (float)m_lambda_c / 8.0f / (1.0f + m_alpha);

    low_a[0] = 0;
    low_a[1] = 0;
    high_a[0] = 0;
    high_a[1] = 0;
}

///
/// \brief EulerianMA::~EulerianMA
///
EulerianMA::~EulerianMA()
{
    Release();
}

///
/// \brief EulerianMA::IsInitialized
/// \return
///
bool EulerianMA::IsInitialized() const
{
    return low_b && high_b;
}

///
/// \brief EulerianMA::GetSize
/// \return
///
cv::Size EulerianMA::GetSize() const
{
	return m_pyr.empty() ? cv::Size(0, 0) : m_pyr[0].size();
}

///
/// \brief EulerianMA::Init
/// \param alpha
/// \param lambda_c
/// \param fl
/// \param fh
/// \param samplingRate
/// \param chromAttenuation
///
void EulerianMA::Init(
	const cv::UMat& rgbframe,
        int alpha,
        int lambda_c,
        float fl,
        float fh,
        int samplingRate,
        float chromAttenuation
        )
{
    Release();

    m_alpha = alpha;
    m_lambda_c = lambda_c;
    m_chromAttenuation = chromAttenuation;

    cv::Mat ntscFrame(rgbframe.size(), CV_32FC3);
    rgb2ntsc(rgbframe.getMat(cv::ACCESS_READ), ntscFrame);

    lappyr(ntscFrame, 0, m_pyr, m_pind);

	for (int i = 0; i < m_pyr.size(); ++i)
	{
		m_lowpass1.push_back(m_pyr[i].clone());
		m_lowpass2.push_back(m_pyr[i].clone());
	}
    m_pyrPrev = m_pyr;

    // --- PREPARATION OF FILTERS COEFFS ------------------------------
    high_b = dcof_bwlp( 1, fh / (float)samplingRate);
    int *ccof_h = ccof_bwlp( 1 );
    float sf_h = sf_bwlp( 1, fh / (float)samplingRate);
    for (int i = 0; i <= 1; ++i)
    {
        high_a[i] = (float)ccof_h[i]*sf_h;
    }
    low_b = dcof_bwlp( 1, fl / (float)samplingRate);
    int *ccof_l = ccof_bwlp( 1 );
    float sf_l = sf_bwlp( 1, fl / (float)samplingRate);
    for (int i = 0; i <= 1; ++i)
    {
        low_a[i] = (float)ccof_l[i] * sf_l;
    }
    // --- PREPARATION OF FILTERS COEFFS ------------------------------


    //  amplify each spatial frequency bands according to Figure 6 of et al. paper
    m_delta = (float)lambda_c / 8.0f / (1.0f + alpha);
    // the factor to boost alpha above the bound et al. have in the paper (for better visualization)
    m_exaggeration_factor = 2.0f;
}

///
/// \brief EulerianMA::Release
///
void EulerianMA::Release()
{
    delete high_b;
    high_b = nullptr;
    delete low_b;
    low_b = nullptr;

	m_pyr.clear();
	m_pyrPrev.clear();
	m_lowpass1.clear();
	m_lowpass2.clear();
}

///
/// \brief EulerianMA::Process
/// \param rgbframe
/// \return
///
cv::UMat EulerianMA::Process(const cv::UMat& rgbframe)
{
    int nLevels = static_cast<int>(m_pyr.size());

	if (rgbframe.size() != m_ntscFrame.size())
	{
		m_ntscFrame.create(rgbframe.size(), CV_32FC3);
	}
    rgb2ntsc(rgbframe.getMat(cv::ACCESS_READ), m_ntscFrame);

    lappyr(m_ntscFrame, 0, m_pyr, m_pind);

    // temporal filtering
#if 1
	TemporalFilter(m_lowpass1, high_a, high_b);
	TemporalFilter(m_lowpass2, low_a, low_b);
#else
	m_lowpass1 = -high_b[1] * m_lowpass1 + high_a[0] * m_pyr + high_a[1] * m_pyrPrev;
    m_lowpass1 = m_lowpass1 / high_b[0];
    m_lowpass2 = -low_b[1] * m_lowpass2 + low_a[0] * m_pyr + low_a[1] * m_pyrPrev;
    m_lowpass2 = m_lowpass2 / low_b[0];
#endif
	m_filtered = (m_lowpass1 - m_lowpass2);

    std::swap(m_pyr, m_pyrPrev);

    // compute the representative wavelength lambda for the lowest spatial frequency band of Laplacian pyramid
    float lambda = sqrt(float(m_ntscFrame.rows * m_ntscFrame.rows + m_ntscFrame.cols * m_ntscFrame.cols)) / 3.0f;  // 3 is experimental constant

    for (int l = nLevels - 1; l >=0; l--)
    {
        // indices = ind - prod(pind(l, :))+1:ind;
        // no need to calc indices
        // because the et al.'s matlab code build pyr stack
        // in one dimension points vec
        // while we are use std::vector container for each level
        /* compute modified alpha for this level */
        float currAlpha = (float)lambda / m_delta / 8.0f - 1.0f;
        currAlpha *= m_exaggeration_factor;

        if (l == nLevels - 1 || l == 0)    // ignore the highest and lowest frequency band
        {
            m_filtered[l] = 0;
        }
        else if (currAlpha > m_alpha)    // representative lambda exceeds
        {
            m_filtered[l] *= m_alpha;
        }
        else
        {
            m_filtered[l] *= currAlpha;
        }
        // go one level down on pyramid, representative lambda will reduce by factor of 2
        lambda /= 2.0;
    }

    // Render on the input video
    reconLpyr(m_filtered, m_output);

	ntsc2rgb(m_output, m_chromAttenuation, m_ntscFrame);

    return m_output.getUMat(cv::ACCESS_READ);
}

///
/// \brief EulerianMA::TemporalFilter
/// \param rgbframe
/// \return
///
void EulerianMA::TemporalFilter(std::vector<cv::Mat>& lowPass,
	const float* coeff_a,
	const float* coeff_b)
{
	const float cfb_1 = 1.0f / coeff_b[0];

	const int pyrSize = static_cast<int>(lowPass.size());
	const int cnls = lowPass[0].channels();
	
#pragma omp parallel for
	for (int i = 0; i < pyrSize; ++i)
	{
		cv::Mat& lpMat = lowPass[i];
		const cv::Mat& currMat = m_pyr[i];
		const cv::Mat& prevMat = m_pyrPrev[i];

		const int height = lpMat.rows;
		const int width = lpMat.cols;

		for (int y = 0; y < height; ++y)
		{
			float* lp = lpMat.ptr<float>(y);
			const float* curr = currMat.ptr<float>(y);
			const float* prev = prevMat.ptr<float>(y);
			for (int x = 0; x < width; ++x)
			{
				for (int cn = 0; cn < cnls; ++cn)
				{
					lp[cn] = cfb_1 * (-coeff_b[1] * lp[cn] + coeff_a[0] * curr[cn] + coeff_a[1] * prev[cn]);
				}
				lp += cnls;
				curr += cnls;
				prev += cnls;
			}
		}
	}
}
