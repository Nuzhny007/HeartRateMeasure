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
void rgb2ntsc(cv::Mat& src, cv::Mat& dst)
{
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec3f i = src.at<cv::Vec3f>(y, x);
            dst.at<cv::Vec3f>(y, x).val[2] = (0.299 * i.val[2] + 0.587 * i.val[1] + 0.114 * i.val[0]) ;
            dst.at<cv::Vec3f>(y, x).val[1] = (0.595716 * i.val[2] - 0.274453 * i.val[1] - 0.321263 * i.val[0]);
            dst.at<cv::Vec3f>(y, x).val[0] = (0.211456 * i.val[2] - 0.522591 * i.val[1] + 0.311135 * i.val[0]);
        }
    }
}

///
/// \brief ntsc2rgb
/// YIQ 2 RGB convert
/// \param src
/// \param dst
///
void ntsc2rgb(cv::Mat& src, cv::Mat& dst)
{
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec3f i = src.at<cv::Vec3f>(y, x);
            dst.at<cv::Vec3f>(y, x).val[2] = (1.0 * i.val[2] + 0.9563 * i.val[1] + 0.621 * i.val[0]);
            dst.at<cv::Vec3f>(y, x).val[1] = (1.0 * i.val[2] - 0.2721 * i.val[1] - 0.6474 * i.val[0]);
            dst.at<cv::Vec3f>(y, x).val[0] = (1.0 * i.val[2] - 1.107 * i.val[1] + 1.7046 * i.val[0]);
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
void lappyr(cv::Mat& src, int level, std::vector<cv::Mat>& lap_arr, std::vector<cv::Size>& nind)
{
    lap_arr.clear();
    cv::Mat down, up, dst;
    cv::Mat tmp=src.clone();

    int max_ht = maxPyrHt(src.size(), cv::Size(4,4));

    if (level == 0) // 'auto' full pyr stack
    {
        level = max_ht;
    }

    for (int l = 0; l < level - 1; l++)
    {
        cv::pyrDown(tmp, down);
        cv::pyrUp(down, up, tmp.size());
        dst = tmp - up;
        lap_arr.push_back(dst);
        nind.push_back(dst.size());
        tmp = down;
        swap(tmp,down);
        if (down.rows < 2)
        {
            std::cout << "Laplacian pyramid stop at level" << l << ", due to the image size" <<  std::endl;
        }
    }

    lap_arr.push_back(down);
    nind.push_back(down.size());
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
    for (int i = lpyr.size() - 2; i >= 0; i--)
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
        //*it *= f;
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
    float factor = 1.0/f;
    for (std::vector<cv::Mat >::const_iterator it = foo.begin(); it != foo.end(); it++)
    {
        //*it /= f;
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
    std::vector<cv::Mat > r;
    for (std::vector<cv::Mat >::const_iterator ita = a.begin(), itb = b.begin(); ita != a.end(); ita++, itb++)
    {
        //*ita -= *itb;
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
        //*ita += *itb;
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
    m_delta = (double)m_lambda_c / 8.0 / (1.0 + m_alpha);

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
/// \brief EulerianMA::Init
/// \param alpha
/// \param lambda_c
/// \param fl
/// \param fh
/// \param samplingRate
/// \param chromAttenuation
///
void EulerianMA::Init(
        cv::Mat rgbframe,
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

    cv::Mat frame;
    rgbframe.convertTo(frame, CV_32FC3, 1.0 / 255.0);
    rgb2ntsc(frame, frame);

    lappyr(frame, 0, pyr, pind);

    //	for(int i=0;i<pyr.size();++i)
    //	{
    //	lowpass1.push_back(cv::Mat(pyr));
    //	lowpass2.push_back(cv::Mat(pyr));
    //	pyr_prev.push_back(cv::Mat(pyr));
    //	}

    lowpass1 = pyr;
    lowpass2 = pyr;
    pyr_prev = pyr;

    // --- PREPARATION OF FILTERS COEFFS ------------------------------
    high_b = dcof_bwlp( 1, fh/(float)samplingRate);
    int *ccof_h = ccof_bwlp( 1 );
    double sf_h = sf_bwlp( 1, fh/(float)samplingRate);
    for(int i = 0; i <= 1; ++i)
    {
        high_a[i]=(double)ccof_h[i]*sf_h;
    }
    low_b = dcof_bwlp( 1, fl/(float)samplingRate);
    int *ccof_l = ccof_bwlp( 1 );
    double sf_l = sf_bwlp( 1, fl/(float)samplingRate);
    for(int i = 0; i <= 1; ++i)
    {
        low_a[i]=(double)ccof_l[i]*sf_l;
    }
    // --- PREPARATION OF FILTERS COEFFS ------------------------------


    //  amplify each spatial frequency bands according to Figure 6 of et al. paper */
    m_delta = (double)lambda_c / 8.0 / (1.0 + alpha);
    // the factor to boost alpha above the bound et al. have in the paper (for better visualization) */
    m_exaggeration_factor = 2.0;
}

///
/// \brief EulerianMA::Release
///
void EulerianMA::Release()
{
    delete high_b;
    delete low_b;
}

///
/// \brief EulerianMA::Process
/// \param rgbframe
/// \return
///
cv::Mat EulerianMA::Process(cv::Mat rgbframe)
{
    int nLevels = pyr.size();

    std::vector<cv::Mat> filtered;

    cv::Mat frame;
    rgbframe.convertTo(frame,CV_32FC3, 1.0 / 255.0);
    rgb2ntsc(frame, frame);

    lappyr(frame, 0, pyr, pind);

    // temporal filtering
    lowpass1 = -high_b[1] * lowpass1 + high_a[0] * pyr + high_a[1] * pyr_prev;
    lowpass1 = lowpass1 / high_b[0];
    lowpass2 = -low_b[1] * lowpass2 + low_a[0] * pyr + low_a[1] * pyr_prev;
    lowpass2 = lowpass2 / low_b[0];
    filtered = (lowpass1 - lowpass2);

    std::swap(pyr, pyr_prev);

    /* compute the representative wavelength lambda for the lowest spatial frequency
    band of Laplacian pyramid */
    double lambda = sqrt(double(frame.rows * frame.rows + frame.cols * frame.cols)) / 3.0;  // 3 is experimental constant

    for (int l = nLevels - 1; l >=0; l--)
    {
        // indices = ind - prod(pind(l, :))+1:ind;
        // no need to calc indices
        // because the et al.'s matlab code build pyr stack
        // in one dimension points vec
        // while we are use std::vector container for each level
        /* compute modified alpha for this level */
        double currAlpha = (double)lambda / m_delta / 8.0 - 1.0;
        currAlpha *= m_exaggeration_factor;

        if (l == nLevels - 1 || l == 0)    // ignore the highest and lowest frequency band
        {
            filtered[l] = 0;
        }
        else if (currAlpha > m_alpha)    // representative lambda exceeds
        {
            filtered[l] *= m_alpha;
        }
        else
        {
            filtered[l] *= currAlpha;
        }
        /* go one level down on pyramid,
        representative lambda will reduce by factor of 2 */
        lambda /= 2.0;
    }

    /* Render on the input video */
    cv::Mat output;
    reconLpyr(filtered, output);


    std::vector<cv::Mat> ch;
    cv::split(output, ch);
    ch[1] *= m_chromAttenuation;
    ch[2] *= m_chromAttenuation;
    cv::merge(ch, output);

    output += frame;
    ntsc2rgb(output, output);

    for (int row = 0; row < output.rows; row++)
    {
        for (int col = 0; col < output.cols; col++)
        {
            cv::Vec3f *p = &output.at<cv::Vec3f>(row, col);
            for (int c = 0; c < 3; c++)
            {
                if (p->val[c] > 1)
                {
                    p->val[c] = 1;
                }
                if (p->val[c] < 0)
                {
                    p->val[c] = 0;
                }
            }
        }
    }
    return output;
}
