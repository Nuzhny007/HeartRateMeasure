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
    float sum = std::accumulate(v.begin(), v.end(), 0.0) / (float)v.size();
    return sum;
}

///
/// \brief LKTracker::LKTracker
/// \param initRegion
///
LKTracker::LKTracker()
{
    term_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.003);
    window_size = cv::Size(8, 8);
    level = 5;
    lambda = 0.5;
    lost = false;
}

///
/// \brief LKTracker::LKTracker
/// \param initRegion
///
LKTracker::LKTracker(cv::Rect initRegion)
{
    bb1 = initRegion;
    bb2 = initRegion;
    BbPoints(points1, bb1);
    BbPoints(points2, bb2);
    term_criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.003);
    window_size = cv::Size(8,8);
    level = 5;
    lambda = 0.5;
    lost = false;
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
        bb1 = initRegion;
        bb2 = initRegion;

        BbPoints(points1, bb1);
        BbPoints(points2, bb2);
    }
    else
    {
        points1.assign(std::begin(points), std::end(points));
        points2.assign(std::begin(points), std::end(points));

        bb2 = cv::boundingRect(points2);
        bb1 = bb2;
    }
    lost = false;
}

///
/// \brief LKTracker::Trackf2f
/// \param img1
/// \param img2
/// \param points1
/// \param points2
/// \return
///
bool LKTracker::Trackf2f(
        const cv::Mat& img1,
        const cv::Mat& img2,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2
        )
{
    //cv::Size subPixWinSize(3, 3);
    //cv::cornerSubPix(img1, points1, subPixWinSize, cv::Size(-1,-1), term_criteria);
    //cv::cornerSubPix(img2, points2, subPixWinSize, cv::Size(-1,-1), term_criteria);

    //Forward-Backward tracking
    cv::calcOpticalFlowPyrLK(img1, img2, points1, points2, status, similarity, window_size, level, term_criteria, lambda, 0);
    cv::calcOpticalFlowPyrLK(img2, img1, points2, pointsFB, FB_status, FB_error, window_size, level, term_criteria, lambda, 0);
    //Compute the real FB-error
    for (size_t i= 0; i < points1.size(); ++i)
    {
        FB_error[i] = norm(pointsFB[i] - points1[i]);
    }
    //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    NormCrossCorrelation(img1, img2, points1, points2);
    return FilterPts(points1, points2);
}

///
/// \brief LKTracker::NormCrossCorrelation
/// \param img1
/// \param img2
/// \param points1
/// \param points2
///
void LKTracker::NormCrossCorrelation(
        const cv::Mat& img1,
        const cv::Mat& img2,
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2
        )
{
    int half_size = 5;
    int size = half_size * 2 + 1;

    for (size_t i = 0; i < points1.size(); ++i)
    {
        if (status[i] == 1 &&
                points1[i].x-half_size>0 &&
                points2[i].x-half_size>0 &&

                points1[i].y-half_size>0 &&
                points2[i].y-half_size>0 &&

                points1[i].x+half_size<img1.cols-1 &&
                points2[i].x+half_size<img2.cols-1 &&

                points1[i].y+half_size<img1.rows-1 &&
                points2[i].y+half_size<img2.rows-1)
        {
            cv::Mat rec0 = img1(cv::Rect(points1[i].x - half_size,points1[i].y - half_size, size, size));
            cv::Mat rec1 = img2(cv::Rect(points2[i].x - half_size,points2[i].y - half_size, size, size));

            cv::Mat res;
            cv::matchTemplate(rec0,rec1, res, cv::TM_CCOEFF_NORMED);
            similarity[i] = ((float *)(res.data))[0];
        }
        else
        {
            similarity[i] = 0.0;
        }
    }
}

///
/// \brief LKTracker::FilterPts
/// \param points1
/// \param points2
/// \return
///
bool LKTracker::FilterPts(
        std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2
        )
{
    //Get Error Medians
    simmed = median(similarity);
    size_t k = 0;
    for (size_t i = 0; i < points2.size(); ++i)
    {
        if (!status[i])
        {
            continue;
        }
        if (similarity[i] >= simmed)
        {
            points1[k] = points1[i];
            points2[k] = points2[i];
            FB_error[k] = FB_error[i];
            k++;
        }
    }
    if (k==0)
    {
        return false;
    }
    points1.resize(k);
    points2.resize(k);
    FB_error.resize(k);

    fbmed = median(FB_error);

    k = 0;
    for (size_t i = 0; i < points2.size(); ++i)
    {
        if(FB_error[i] <= fbmed)
        {
            points1[k] = points1[i];
            points2[k] = points2[i];
            k++;
        }
    }
    points1.resize(k);
    points2.resize(k);
    if (k > 30)
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
    if (imgPrev.size() != img.size())
    {
        img.copyTo(imgPrev);
        return;
    }

    if (points1.size() < 70)
    {
        points1.clear();
        // Generate point set
        BbPoints(points1, bb2);
    }
    // Если точек нет, трекер сорвался
    if (points1.size() < 1)
    {
        tvalid = false;
        tracked = false;
        return;
    }

    //Frame-to-frame tracking with forward-backward error cheking
    tracked = Trackf2f(imgPrev, img, points1, points2);

    if (tracked)
    {
        //Bounding box prediction
        BbPredict(points1, points2);
        if (getFB() > 10 || bb2.x > img.cols || bb2.y > img.rows || bb2.br().x < 1 || bb2.br().y < 1)
        {
            tvalid = false; //too unstable prediction or bounding box out of image
            tracked = false;
            printf("Too unstable predictions FB error=%f\n",getFB());
            return;
        }

        tvalid =true;
    }
    else
    {
        printf("No points tracked\n");
    }
    bb1 = bb2;

    if(tracked == false)
    {
        lost = true;
    }
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
    int stepx = ceilf((float)(bb.width-2.0*margin_h)/(float)max_pts);
    int stepy = ceilf((float)(bb.height-2.0*margin_v)/(float)max_pts);

    points.clear();
    points.reserve(max_pts * max_pts);

    for (int y = bb.y + margin_v; y < bb.y + bb.height - margin_v; y += stepy)
    {
        for (int x = bb.x + margin_h; x < bb.x + bb.width - margin_h; x += stepx)
        {
            points.push_back(cv::Point2f(x,y));
        }
    }
}

///
/// \brief LKTracker::BbPredict
/// Прогноз изменения положения и масштаба рамки по изменению координат трекаемых внутри нее точек
/// \param points1
/// \param points2
///
void LKTracker::BbPredict(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2
        )
{
    int npoints = points1.size();
    std::vector<float> xoff(npoints);
    std::vector<float> yoff(npoints);
    std::cout << npoints << " points tracked." << std::endl;

    for (int i=0; i<npoints; i++)
    {
        xoff[i]=points2[i].x-points1[i].x;
        yoff[i]=points2[i].y-points1[i].y;
    }
    float dx = median(xoff);
    float dy = median(yoff);
    float s;
    if (npoints > 1)
    {
        std::vector<float> d(npoints*(npoints-1)/2);
        int ind=0;

        for (int i=0; i<npoints; i++)
        {
            for (int j=i+1; j<npoints; j++)
            {
                d[ind] = norm(points2[i]-points2[j]) / norm(points1[i]-points1[j]);
                ++ind;
            }
        }

        s = median(d);
        d.clear();
    }
    else
    {
        s = 1.0;
    }
    float s1 = 0.5*(s-1)*(float)bb1.width;
    float s2 = 0.5*(s-1)*(float)bb1.height;
    ////printf("s= %f s1= %f s2= %f \n",s,s1,s2);
    bb2.x = cvRound( (float)bb1.x + dx - s1);
    bb2.y = cvRound( (float)bb1.y + dy - s2);
    bb2.width = cvRound((float)bb1.width*s);
    bb2.height = cvRound((float)bb1.height*s);
    std::cout << "Predicted box: " << bb2 << std::endl;
}
