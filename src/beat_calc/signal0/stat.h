#pragma once

#include <array>
#include <deque>
#include <opencv2/opencv.hpp>

#include "../plugin.h"

///
/// \brief The Measure class
/// Измерение.
/// Включает в себя момент времени в который произведено измерение и собствнно само измерение.
///
template <typename VAL>
class Measure
{
public:
    int64 t; // Время измерения
    VAL val;		  // Значение измеренной величины

    Measure (int64 t_, const VAL& val_)
    {
        t = t_;
        val = val_;
    }
};

///
/// \brief sqr
/// \param v
/// \return square of the value
///
template<typename T>
T sqr(T v)
{
    return v * v;
}

///
/// \brief The Gaussian class
///
template<typename MEAS_T, typename DATA_T>
class Gaussian
{
public:
    ///
    /// \brief Gaussian
    /// \param eps
    /// \param alpha
    ///
    Gaussian(DATA_T eps, DATA_T alpha, DATA_T defVariance, DATA_T minVariance, DATA_T maxVariance)
        :
          m_mean(0),
          m_var(defVariance),
          m_minVar(minVariance),
          m_maxVar(maxVariance),
          m_eps(eps),
          m_alpha(alpha),
          m_lastMeasure(0),
          m_measuresCount(0)
    {
    }

    ///
    /// \brief Gaussian
    /// \param measure
    /// \param eps
    /// \param alpha
    ///
    Gaussian(MEAS_T measure, DATA_T eps, DATA_T alpha, DATA_T defVariance, DATA_T minVariance, DATA_T maxVariance)
        :
          m_mean(measure),
          m_var(defVariance),
          m_minVar(minVariance),
          m_maxVar(maxVariance),
          m_eps(eps),
          m_alpha(alpha),
          m_lastMeasure(measure),
          m_measuresCount(1)
    {
    }

    ///
    /// \brief AddMeasure
    /// \param measure
    /// \return
    ///
    bool AddMeasure(MEAS_T measure)
    {
        if (CheckMeasure(measure))
        {
            m_lastMeasure = measure;
            ++m_measuresCount;
            UpdateModel(measure);
            return true;
        }
        else
        {
            return false;
        }
    }

    ///
    /// \brief CurrValue
    /// \return
    ///
    MEAS_T CurrValue() const
    {
        return static_cast<MEAS_T>(m_mean);
    }

    ///
    /// \brief CurrValue
    /// \return
    ///
    MEAS_T Epsilon() const
    {
        return static_cast<MEAS_T>(m_eps * m_var);
    }

private:
    ///
    /// \brief m_mean
    /// Mean value of the Gaussian
    ///
    DATA_T m_mean;
    ///
    /// \brief m_var
    /// Variance
    ///
    DATA_T m_var;
    ///
    /// \brief m_maxVar
    /// Minimum value of Variance
    ///
    DATA_T m_minVar;
    ///
    /// \brief m_maxVar
    /// Maximum value of Variance
    ///
    DATA_T m_maxVar;
    ///
    /// \brief m_eps
    /// Model accuracy:
    /// The maximum ratio of the deviation of the current measure from its average value to the mean square deviation
    /// The value 2.7 ~= 0.95 probablity
    ///
    DATA_T m_eps;

    ///
    /// \brief m_alpha
    /// Exponential smoothing coefficient
    ///
    DATA_T m_alpha;

    ///
    /// \brief m_lastMeasure
    /// Latest measure
    ///
    MEAS_T m_lastMeasure;
    size_t m_measuresCount;

    ///
    /// \brief CheckMeasure
    /// \param measure
    /// \return
    ///
    bool CheckMeasure(MEAS_T measure) const
    {
        return m_eps * m_var > std::abs(m_mean - measure);
    }
    ///
    /// \brief UpdateModel
    /// \param measure
    ///
    void UpdateModel(MEAS_T measure)
    {
        std::cout << "Update model (" << m_eps << ", " << m_alpha << "):" << std::endl;
        std::cout << "Old: Mean = " << m_mean << ", Var = " << m_var << std::endl;

        m_var = sqrt((1 - m_alpha) * sqr(m_var) + m_alpha * sqr(measure - m_mean));
        if (m_var < m_minVar)
        {
            m_var = m_minVar;
        }
        else if (m_var > m_maxVar)
        {
            m_var = m_maxVar;
        }
        m_mean = (1 - m_alpha) * m_mean + m_alpha * measure;

        std::cout << "New: Mean = " << m_mean << ", Var = " << m_var << std::endl;
    }
};

///
/// \brief The WeightedGaussian class
///
template<typename MEAS_T, typename DATA_T>
class WeightedGaussian : public Gaussian<MEAS_T, DATA_T>
{
public:
    ///
    /// \brief WeightedGaussian
    /// \param eps
    /// \param alpha
    ///
    WeightedGaussian(DATA_T eps = 2.7, DATA_T alpha = 0.1, DATA_T defVariance = 5, DATA_T minVariance = 2, DATA_T maxVariance = 10, DATA_T procAlpha = 0.05)
        :
          Gaussian<MEAS_T, DATA_T>(eps, alpha, defVariance, minVariance, maxVariance),
          m_weight(0),
          m_procAlpha(procAlpha)
    {
    }

    ///
    /// \brief WeightedGaussian
    /// \param measure
    /// \param eps
    /// \param alpha
    ///
    WeightedGaussian(MEAS_T measure, DATA_T eps = 2.7, DATA_T alpha = 0.1, DATA_T defVariance = 5, DATA_T minVariance = 2, DATA_T maxVariance = 10, DATA_T procAlpha = 0.05)
        :
          Gaussian<MEAS_T, DATA_T>(measure, eps, alpha, defVariance, minVariance, maxVariance),
          m_weight(0),
          m_procAlpha(procAlpha)
    {
    }

    ///
    /// \brief Weight
    /// \return
    ///
    DATA_T Weight() const
    {
        return m_weight;
    }

    ///
    /// \brief UpdateWeight
    /// \param increase
    ///
    void UpdateWeight(bool increase)
    {
        m_weight = (1 - m_procAlpha) * m_weight + (increase ? m_procAlpha : 0);
    }

private:
    ///
    /// \brief m_weight
    /// Weight of the process
    ///
    DATA_T m_weight;

    ///
    /// \brief m_procAlpha
    ///
    DATA_T m_procAlpha;
};

///
/// \brief The GaussMixture class
///
template<size_t GAUSS_COUNT, typename MEAS_T, typename DATA_T>
class GaussMixture
{
public:
    GaussMixture(DATA_T defaultVariance, DATA_T minVariance, DATA_T maxVariance,
                 DATA_T defaultGaussEps, DATA_T defaultGaussAlpha,
                 DATA_T defaultProcAlpha, DATA_T weightThreshold)
        :
          m_currProc(0),
		  m_prevProc(0),
		  m_procChangeCounter(0),
          m_createdProcesses(0),
          m_weightThreshold(weightThreshold),
          m_defaultGaussEps(defaultGaussEps),
          m_defaultGaussAlpha(defaultGaussAlpha),
          m_defaultProcAlpha(defaultProcAlpha),
          m_defaultVariance(defaultVariance),
          m_minVariance(minVariance),
          m_maxVariance(maxVariance),
          m_timeStamp(0)
    {
    }

    ///
    /// \brief Reset
    ///
    void Reset()
    {
        m_currProc = 0;
		m_prevProc = 0;
		m_procChangeCounter = 0;
        m_createdProcesses = 0;
        m_timeStamp = 0;
        for (auto& hist : m_history)
        {
            hist.clear();
        }
    }

    ///
    /// \brief AddMeasure
    /// \param measure
    /// \return
    ///
    bool AddMeasure(MEAS_T measure)
    {
        //std::cout << "--------------------------------------------" << std::endl;
        //std::cout << "Measure = " << measure << std::endl;

        bool findProcess = false;
		
		if (m_createdProcesses)
		{
			if (m_procList[m_currProc].AddMeasure(measure))
			{
				//std::cout << "Measure added to the current process " << m_currProc << std::endl;
				findProcess = true;
			}
			else
			{
				for (size_t i = 0; i < m_createdProcesses; ++i)
				{
					if (m_procList[i].AddMeasure(measure))
					{
						//std::cout << "Measure added to the process " << i << std::endl;

						m_currProc = i;
						findProcess = true;
						break;
					}
				}
			}
		}

        if (!findProcess)
        {
            if (m_createdProcesses < GAUSS_COUNT)
            {
                ++m_createdProcesses;
                m_currProc = m_createdProcesses - 1;

                m_procList[m_currProc] = WeightedGaussian<MEAS_T, DATA_T>(measure, m_defaultGaussEps, m_defaultGaussAlpha, m_defaultVariance, m_minVariance, m_maxVariance, m_defaultProcAlpha);
                findProcess = true;

                //std::cout << "Create new process " << m_createdProcesses << std::endl;
            }
            else
            {
                auto minWeight = m_procList[0].Weight();
                size_t minProc = 0;
                for (size_t i = 1; i < m_createdProcesses; ++i)
                {
                    if (i != m_prevProc && m_procList[i].Weight() < minWeight)
                    {
                        minProc = i;
                        minWeight = m_procList[i].Weight();
                    }
                }

                m_currProc = minProc;
                m_procList[m_currProc] = WeightedGaussian<MEAS_T, DATA_T>(measure, m_defaultGaussEps, m_defaultGaussAlpha, m_defaultVariance, m_minVariance, m_maxVariance, m_defaultProcAlpha);

                //std::cout << "Create new process from " << minProc << " with weight " << minWeight << std::endl;
            }
        }

        //std::cout << "Update weights:" << std::endl;
        for (size_t i = 0; i < m_createdProcesses; ++i)
        {
            m_procList[i].UpdateWeight(i == m_currProc);

            //std::cout << ((i == m_currProc) ? "+ " : "- ") << m_procList[i].CurrValue() << ": " << m_procList[i].Weight() << " - " << m_weightThreshold << std::endl;

            m_history[i].push_back(HistoryVal(m_procList[i].CurrValue(), m_procList[i].Epsilon(), m_procList[i].Weight(), m_timeStamp));
            if (m_history[i].size() > MAX_HISTORY)
            {
                m_history[i].pop_front();
            }
        }

        //std::cout << "--------------------------------------------" << std::endl;

        ++m_timeStamp;
		if (m_prevProc != m_currProc)
		{
			++m_procChangeCounter;
			if (m_procChangeCounter > 25)
			{
				m_prevProc = m_currProc;
				m_procChangeCounter = 0;
			}
		}
		else
		{
			m_procChangeCounter = 0;
		}

        return m_procList[m_currProc].Weight() > m_weightThreshold;
    }

    ///
    /// \brief CurrValue
    /// \return
    ///
    MEAS_T CurrValue() const
    {
        return m_procList[m_prevProc].CurrValue();
    }

    ///
    /// \brief AllValues
    /// \param vals
    ///
    void AllValues(std::vector<MEAS_T>& vals) const
    {
        vals.resize(m_createdProcesses);

        for (size_t i = 0; i < m_createdProcesses; ++i)
        {
            vals[i] = m_procList[i].CurrValue();
        }
    }

    ///
    /// \brief RobustValues
    /// \param vals
    ///
    void RobustValues(std::vector<MEAS_T>& vals) const
    {
        vals.clear();

        for (size_t i = 0; i < m_createdProcesses; ++i)
        {
            if (m_procList[i].Weight() > m_weightThreshold)
            {
                vals.push_back(m_procList[i].CurrValue());
            }
        }
    }

    ///
    /// \brief Visualize
    ///
    void Visualize(bool saveResult, int frameInd, const std::string& wndName)
    {
        const int oneHeight = 150;
        const DATA_T maxVal = 200;

        cv::Mat img((oneHeight + 1) * GAUSS_COUNT, MAX_HISTORY, CV_8UC3, cv::Scalar(255, 255, 255));

        for (int i = 0; i < static_cast<int>(GAUSS_COUNT); ++i)
        {
            if (i)
            {
                cv::line(img, cv::Point(0, (oneHeight + 1) * i - 1), cv::Point(img.cols - 1, (oneHeight + 1) * i - 1), cv::Scalar(0, 0, 0));
            }

            const auto& hist = m_history[i];

            for (int x = 0, stop = static_cast<int>(hist.size()); x < stop; ++x)
            {
                int ts = 0;

                if (MAX_HISTORY > m_timeStamp)
                {
                    ts = hist[x].m_timeStamp;
                }
                else
                {
                    ts = MAX_HISTORY - m_timeStamp + hist[x].m_timeStamp;
                }

                if (ts < 0)
                {
                    break;
                }

                // Background
                cv::Scalar backColor = (hist[x].m_weight > m_weightThreshold) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::Rect backRect(ts, (oneHeight + 1) * i, 1, oneHeight);
                cv::rectangle(img, backRect, backColor, -1, cv::LINE_8, 0);

                // Values
                int val = cvRound((oneHeight * hist[x].m_mean) / maxVal);
                int upVal = cvRound((oneHeight * (hist[x].m_mean + hist[x].m_var)) / maxVal);
                int loVal = cvRound((oneHeight * (hist[x].m_mean - hist[x].m_var)) / maxVal);
                cv::circle(img, cv::Point(ts, backRect.y + (oneHeight - val)), 1, cv::Scalar(255, 0, 255), -1);
                cv::circle(img, cv::Point(ts, backRect.y + (oneHeight - upVal)), 1, cv::Scalar(255, 0, 0), -1);
                cv::circle(img, cv::Point(ts, backRect.y + (oneHeight - loVal)), 1, cv::Scalar(255, 0, 0), -1);

                if (x == stop - 1)
                {
                    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
                    double fontScale = 0.7;
                    cv::putText(img, std::to_string(hist[x].m_mean), cv::Point(MAX_HISTORY / 2, backRect.y + 20), fontFace, fontScale, cv::Scalar(0, 0, 0), 2);
                }
            }
        }

        cv::imshow("mixture " + wndName, img);

        if (saveResult)
        {
            std::string fileName = "mixture_" + wndName + "/" + std::to_string(frameInd) + ".png";
            cv::imwrite(fileName, img);
        }
    }

private:
    ///
    /// \brief m_procList
    ///
    std::array<WeightedGaussian<MEAS_T, DATA_T>, GAUSS_COUNT> m_procList;
    ///
    /// \brief m_currProc
    ///
    size_t m_currProc = 0;
	///
	/// \brief m_prevProc
	///
	size_t m_prevProc = 0;
	///
	/// \brief m_prevProc
	///
	size_t m_procChangeCounter = 0;
    ///
    /// \brief m_createdProcesses
    ///
    size_t m_createdProcesses = 1;

    ///
    /// \brief m_weightThreshold
    ///
    DATA_T m_weightThreshold;

    ///
    /// \brief m_defaultGaussEps
    ///
    DATA_T m_defaultGaussEps;
    ///
    /// \brief m_defaultGaussAlpha
    ///
    DATA_T m_defaultGaussAlpha;

    ///
    /// \brief m_defaultProcAlpha
    ///
    DATA_T m_defaultProcAlpha;
    ///
    /// \brief m_defaultVariance
    ///
    DATA_T m_defaultVariance;
    ///
    /// \brief m_minVariance
    ///
    DATA_T m_minVariance;
    ///
    /// \brief m_maxVariance
    ///
    DATA_T m_maxVariance;

    ///
    /// \brief m_timeStamp
    ///
    int m_timeStamp = 0;

    struct HistoryVal
    {
        DATA_T m_mean;
        DATA_T m_var;
        DATA_T m_weight;
        int m_timeStamp;

        HistoryVal(DATA_T mean, DATA_T var, DATA_T weight, int timeStamp)
            : m_mean(mean), m_var(var), m_weight(weight), m_timeStamp(timeStamp)
        {

        }
    };
    static const int MAX_HISTORY = 800;

    std::deque<HistoryVal> m_history[GAUSS_COUNT];
};
