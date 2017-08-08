#pragma once

#include <array>
#include <opencv2/opencv.hpp>

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
    Gaussian(DATA_T eps, DATA_T alpha)
        :
          m_mean(0),
          m_var(10),
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
    Gaussian(MEAS_T measure, DATA_T eps, DATA_T alpha)
        :
          m_mean(measure),
          m_var(10),
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

    MEAS_T CurrValue() const
    {
        return static_cast<MEAS_T>(m_mean);
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
    WeightedGaussian(DATA_T eps = 2.7, DATA_T alpha = 0.1)
        :
          Gaussian<MEAS_T, DATA_T>(eps, alpha),
          m_weight(0),
          m_procAlpha(0.05)
    {
    }

    ///
    /// \brief WeightedGaussian
    /// \param measure
    /// \param eps
    /// \param alpha
    ///
    WeightedGaussian(MEAS_T measure, DATA_T eps = 2.7, DATA_T alpha = 0.1)
        :
          Gaussian<MEAS_T, DATA_T>(measure, eps, alpha),
          m_weight(0),
          m_procAlpha(0.05)
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
    GaussMixture()
        :
          m_currProc(0),
          m_createdProcesses(1),
          m_weightThreshold(0.2),
          m_defaultGussEps(2.7),
          m_defaultGussAlpha(0.1)
    {
    }

    ///
    /// \brief AddMeasure
    /// \param measure
    /// \return
    ///
    bool AddMeasure(MEAS_T measure)
    {
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Measure = " << measure << std::endl;

        bool findProcess = false;

        if (m_procList[m_currProc].AddMeasure(measure))
        {
            std::cout << "Measure added to the current process " << m_currProc << std::endl;
            findProcess = true;
        }
        else
        {
            for (size_t i = 0; i < m_createdProcesses; ++i)
            {
                if (m_procList[i].AddMeasure(measure))
                {
                    std::cout << "Measure added to the process " << i << std::endl;

                    m_currProc = i;
                    findProcess = true;
                    break;
                }
            }
        }

        if (!findProcess)
        {
            if (m_createdProcesses < GAUSS_COUNT)
            {
                ++m_createdProcesses;
                m_currProc = m_createdProcesses - 1;

                m_procList[m_currProc] = WeightedGaussian<MEAS_T, DATA_T>(measure, m_defaultGussEps, m_defaultGussAlpha);
                findProcess = true;

                std::cout << "Create new process " << m_createdProcesses << std::endl;
            }
            else
            {
                auto minWeight = m_procList[0].Weight();
                size_t minProc = 0;
                for (size_t i = 1; i < m_createdProcesses; ++i)
                {
                    if (m_procList[i].Weight() < minWeight)
                    {
                        minProc = i;
                        minWeight = m_procList[i].Weight();
                    }
                }

                m_currProc = minProc;
                m_procList[m_currProc] = WeightedGaussian<MEAS_T, DATA_T>(measure, m_defaultGussEps, m_defaultGussAlpha);

                std::cout << "Create new process from " << minProc << " with weight " << minWeight << std::endl;
            }
        }

        std::cout << "Update weights:" << std::endl;
        for (size_t i = 0; i < m_createdProcesses; ++i)
        {
            m_procList[i].UpdateWeight(i == m_currProc);

            std::cout << ((i == m_currProc) ? "+ " : "- ") << m_procList[i].CurrValue() << ": " << m_procList[i].Weight() << " - " << m_weightThreshold << std::endl;
        }

        std::cout << "--------------------------------------------" << std::endl;

        return m_procList[m_currProc].Weight() > m_weightThreshold;
    }

    ///
    /// \brief CurrValue
    /// \return
    ///
    MEAS_T CurrValue() const
    {
        return m_procList[m_currProc].CurrValue();
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

private:
    ///
    /// \brief m_procList
    ///
    std::array<WeightedGaussian<MEAS_T, DATA_T>, GAUSS_COUNT> m_procList;
    ///
    /// \brief m_currProc
    ///
    size_t m_currProc;
    ///
    /// \brief m_createdProcesses
    ///
    size_t m_createdProcesses;

    ///
    /// \brief m_weightThreshold
    ///
    DATA_T m_weightThreshold;

    ///
    /// \brief m_defaultGussEps
    ///
    DATA_T m_defaultGussEps;
    ///
    /// \brief m_defaultGussAlpha
    ///
    DATA_T m_defaultGussAlpha;
};
