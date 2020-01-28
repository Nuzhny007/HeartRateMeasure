#include "LKTracker.h"
#include <algorithm>
#include <numeric>

///
/// \brief median
/// Находим среднее
/// \param v
/// \return
///
float median(const std::vector<float>& v)
{
    float sum = std::accumulate(v.begin(), v.end(), 0.0f) / (float)v.size();
    return sum;
}

///
/// \brief LKTracker::LKTracker
/// \param initRegion
///
LKTracker::LKTracker()
{
    m_termCriteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 0.05);
    m_windowSize = cv::Size(8, 8);
    m_level = 5;
    m_lambda = 0.005f;
    m_lost = false;
}

///
/// \brief LKTracker::LKTracker
/// \param initRegion
///
LKTracker::LKTracker(cv::Rect initRegion)
{
    m_bb1 = initRegion;
    m_bb2 = initRegion;
    BbPoints(m_points1, m_bb1);
    BbPoints(m_points2, m_bb2);
    m_termCriteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 0.05);
    m_windowSize = cv::Size(8, 8);
    m_level = 5;
    m_lambda = 0.005f;
    m_lost = false;
}

///
/// \brief LKTracker::ReinitTracker
/// \param initRegion
/// \param points
///
void LKTracker::ReinitTracker(cv::Rect initRegion, const std::vector<cv::Point2f>& points)
{
    if (points.empty())
    {
        m_bb1 = initRegion;
        m_bb2 = initRegion;

        BbPoints(m_points1, m_bb1);
        BbPoints(m_points2, m_bb2);
    }
    else
    {
        m_points1.assign(std::begin(points), std::end(points));
        m_points2.assign(std::begin(points), std::end(points));

        m_bb2 = cv::boundingRect(m_points2);
        m_bb1 = m_bb2;
    }
    m_lost = false;
}

///
/// \brief LKTracker::Trackf2f
/// \param img1
/// \param img2
/// \return
///
bool LKTracker::Trackf2f(
        const cv::Mat& img1,
        const cv::Mat& img2
        )
{
	if (img1.channels() == 1)
	{
		cv::Size subPixWinSize(3, 3);
		cv::cornerSubPix(img1, m_points1, subPixWinSize, cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.05));
		cv::cornerSubPix(img2, m_points2, subPixWinSize, cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.05));
	}

    //Forward-Backward tracking
    cv::calcOpticalFlowPyrLK(img1, img2, m_points1, m_points2, m_status, m_similarity, m_windowSize, m_level, m_termCriteria, 0, m_lambda);
    cv::calcOpticalFlowPyrLK(img2, img1, m_points2, m_pointsFB, m_FBStatus, m_FBError, m_windowSize, m_level, m_termCriteria, 0, m_lambda);
    //Compute the real FB-error
    for (size_t i = 0; i < m_points1.size(); ++i)
    {
        m_FBError[i] = static_cast<float>(norm(m_pointsFB[i] - m_points1[i]));
    }
    //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    NormCrossCorrelation(img1, img2);
    return FilterPts();
}

///
/// \brief LKTracker::NormCrossCorrelation
/// \param img1
/// \param img2
///
void LKTracker::NormCrossCorrelation(
        const cv::Mat& img1,
        const cv::Mat& img2
        )
{
    int half_size = 5;
    int size = half_size * 2 + 1;

	size_t k = 0;
    for (size_t i = 0; i < m_points1.size(); ++i)
    {
		if (!m_status[i])
		{
			continue;
		}

		cv::Point2f& pt1 = m_points1[k];
		cv::Point2f& pt2 = m_points2[k];

		if (pt1.x - half_size > 0 &&
			pt2.x - half_size > 0 &&
			pt1.y - half_size > 0 &&
			pt2.y - half_size > 0 &&
			pt1.x + half_size < img1.cols - 1 &&
			pt2.x + half_size < img2.cols - 1 &&
			pt1.y + half_size < img1.rows - 1 &&
			pt2.y + half_size < img2.rows - 1)
		{
			cv::Mat rec0 = img1(cv::Rect(static_cast<int>(pt1.x - half_size), static_cast<int>(pt1.y - half_size), size, size));
			cv::Mat rec1 = img2(cv::Rect(static_cast<int>(pt2.x - half_size), static_cast<int>(pt2.y - half_size), size, size));

			cv::Mat res;
			cv::matchTemplate(rec0, rec1, res, cv::TM_CCOEFF_NORMED);
			m_similarity[k] = ((float*)(res.data))[0];

			m_points1[k] = m_points1[i];
			m_points2[k] = m_points2[i];
			m_FBError[k] = m_FBError[i];

			++k;
		}
    }
	m_points1.resize(k);
	m_points2.resize(k);
	m_FBError.resize(k);
	m_similarity.resize(k);

	//std::cout << "Points resized to " << k << ", img = " << img1.size() << std::endl;
}

///
/// \brief LKTracker::FilterPts
/// \return
///
bool LKTracker::FilterPts()
{
    //Get Error Medians
    m_simmed = median(m_similarity);
    size_t k = 0;
    for (size_t i = 0; i < m_points2.size(); ++i)
    {
        if (m_similarity[i] >= m_simmed)
        {
            m_points1[k] = m_points1[i];
            m_points2[k] = m_points2[i];
            m_FBError[k] = m_FBError[i];
            k++;
        }
    }
	//std::cout << "Points resized by simmed " << m_simmed << " to " << k << std::endl;
    if (k == 0)
    {
        return false;
    }
    m_points1.resize(k);
    m_points2.resize(k);
    m_FBError.resize(k);

    m_fbmed = median(m_FBError);

    k = 0;
    for (size_t i = 0; i < m_points2.size(); ++i)
    {
        if (m_FBError[i] <= m_fbmed)
        {
            m_points1[k] = m_points1[i];
            m_points2[k] = m_points2[i];
            k++;
        }
    }
    m_points1.resize(k);
    m_points2.resize(k);
    if (k > MinPoints)
    {
        return true;
    }
    else
    {
        return false;
    }
}

///
/// \brief LKTracker::Track
/// Трекер методом оптического потока по набору точек
/// \param img1
/// \param img2
///
void LKTracker::Track(
        cv::Mat img
        )
{
    if (m_imgPrev.size() != img.size())
    {
        img.copyTo(m_imgPrev);
        return;
    }

    if (m_points1.size() < PointsToGenerate)
    {
        m_points1.clear();
        // Generate point set
        BbPoints(m_points1, m_bb2);
    }
    // Если точек нет, трекер сорвался
    if (m_points1.size() < 1)
    {
        m_tvalid = false;
        m_tracked = false;
        return;
    }

    //Frame-to-frame tracking with forward-backward error cheking
    m_tracked = Trackf2f(m_imgPrev, img);

    if (m_tracked)
    {
        //Bounding box prediction
        BbPredict();
        if (m_fbmed > 10 || m_bb2.x > img.cols || m_bb2.y > img.rows || m_bb2.br().x < 1 || m_bb2.br().y < 1)
        {
            m_tvalid = false; //too unstable prediction or bounding box out of image
            m_tracked = false;
            printf("Too unstable predictions FB error=%f\n", m_fbmed);
            return;
        }

        m_tvalid = true;
    }
    else
    {
        printf("No points tracked\n");
    }
    m_bb1 = m_bb2;

    m_lost = (m_tracked == false);
}

///
/// \brief LKTracker::BbPoints
/// Генерируем набор точек
/// \param points
/// \param bb
///
void LKTracker::BbPoints(
        std::vector<cv::Point2f>& points,
        const cv::Rect& bb
        )
{
    int max_pts = 20; // Number of points on each dimention
    int margin_h = 0; // horizontal margin
    int margin_v = 0; // vertial margin
    int stepx = static_cast<int>(ceilf((bb.width - 2.0f * margin_h) / max_pts));
    int stepy = static_cast<int>(ceilf((bb.height - 2.0f * margin_v) / max_pts));

    points.clear();
    points.reserve(max_pts * max_pts);

    for (int y = bb.y + margin_v; y < bb.y + bb.height - margin_v; y += stepy)
    {
        for (int x = bb.x + margin_h; x < bb.x + bb.width - margin_h; x += stepx)
        {
            points.push_back(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
        }
    }
	//std::cout << "Generate " << points.size() << " points" << std::endl;
}

///
/// \brief LKTracker::BbPredict
/// Прогноз изменения положения и масштаба рамки по изменению координат трекаемых внутри нее точек
///
void LKTracker::BbPredict()
{
    int npoints = static_cast<int>(m_points1.size());
    std::vector<float> xoff(npoints);
    std::vector<float> yoff(npoints);
    std::cout << npoints << " points tracked." << std::endl;

    for (int i = 0; i < npoints; i++)
    {
        xoff[i] = m_points2[i].x - m_points1[i].x;
        yoff[i] = m_points2[i].y - m_points1[i].y;
    }
    float dx = median(xoff);
    float dy = median(yoff);
    float s = 1.0;
    if (npoints > 1)
    {
        std::vector<float> d(npoints * (npoints - 1) / 2);
        int ind = 0;

        for (int i = 0; i < npoints; i++)
        {
            for (int j = i + 1; j < npoints; j++)
            {
                d[ind] = static_cast<float>(norm(m_points2[i] - m_points2[j]) / norm(m_points1[i] - m_points1[j]));
                ++ind;
            }
        }

        s = median(d);
     }

    float s1 = 0.5f * (s - 1) * m_bb1.width;
    float s2 = 0.5f * (s - 1) * m_bb1.height;
    ////printf("s= %f s1= %f s2= %f \n",s,s1,s2);
    m_bb2.x = cvRound(m_bb1.x + dx - s1);
    m_bb2.y = cvRound(m_bb1.y + dy - s2);
    m_bb2.width = cvRound(m_bb1.width * s);
    m_bb2.height = cvRound(m_bb1.height * s);
    std::cout << "Predicted box: " << m_bb2 << std::endl;
}

///
/// \brief LKTracker::IsLost
/// \return
///
bool LKTracker::IsLost(void) const
{
    return m_lost;
}

///
/// \brief LKTracker::GetTrackedRegion
/// \return
///
cv::Rect LKTracker::GetTrackedRegion(void) const
{
    return m_bb2;
}

///
/// \brief LKTracker::GetMovingSum
/// \return
///
bool LKTracker::GetMovingSum(cv::Point2d& moveSum) const
{
    moveSum = cv::Point2d(0, 0);

    if (m_points1.size() == m_points2.size() && m_points1.size() > PointsToGenerate)
    {
        for (size_t i = 0; i < m_points1.size(); ++i)
        {
            moveSum.x += m_points1[i].x - m_points2[i].x;
            moveSum.y += m_points1[i].y - m_points2[i].y;
        }

        return true;
    }
    return false;
}

///
/// \brief LKTracker::GetPoints
/// \return
///
void LKTracker::GetPoints(std::vector<cv::Point2f>& points)
{
	points.assign(m_points2.begin(), m_points2.end());
}
