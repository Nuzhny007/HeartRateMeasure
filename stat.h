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
          m_var(0),
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
          m_var(measure),
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
        return m_eps * m_var < std::abs(m_mean - measure);
    }
    ///
    /// \brief UpdateModel
    /// \param measure
    ///
    void UpdateModel(MEAS_T measure)
    {
        m_var = sqrt((1 - m_alpha) * sqr(m_var) + m_alpha * sqr(measure - m_mean));
        m_mean = (1 - m_alpha) * m_mean + m_alpha * measure;
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
    WeightedGaussian(DATA_T eps = 2.7, DATA_T alpha = 0.9)
        :
          Gaussian<MEAS_T, DATA_T>(eps, alpha),
          m_weight(0)
    {
    }

    ///
    /// \brief WeightedGaussian
    /// \param measure
    /// \param eps
    /// \param alpha
    ///
    WeightedGaussian(MEAS_T measure, DATA_T eps = 2.7, DATA_T alpha = 0.9)
        :
          Gaussian<MEAS_T, DATA_T>(measure, eps, alpha),
          m_weight(0)
    {
    }

private:
    ///
    /// \brief m_weight
    ///
    DATA_T m_weight;
};

///
/// \brief The GaussProcess class
///
template<size_t GAUSS_COUNT, typename MEAS_T, typename DATA_T>
class GaussProcess
{
public:
    GaussProcess()
        :
          m_currProc(0),
          m_createdProcesses(1)
    {
    }

    ///
    /// \brief AddMeasure
    /// \param measure
    /// \return
    ///
    void AddMeasure(MEAS_T measure)
    {
        m_procList[m_currProc].AddMeasure(measure);

#if 0
        bool find_process = false;

        for (size_t ind = 0; ind < m_createdProcesses; ++ind)
        {
            if (m_procList[ind].AddMeasure(measure))
            {
                m_currProc = ind;
                find_process = true;
                break;
            }
        }
        if (!find_process) // Процесс не найден
        {
            // Создаём новый процесс или,
            if (created_processes < PROC_PER_PIXEL)
            {
                ++created_processes;
                curr_proc = created_processes - 1;

                proc_list[curr_proc].set_mu_sigma(new_val, min_sigma_val);

                find_process = true;
            }
            // если количество процессов равно PROC_PER_PIXEL, ищем процесс с наименьшим весом
            else
            {
                float_t min_weight = proc_list[0].weight;
                size_t min_proc = 0;
                for (size_t proc_ind = 1; proc_ind < created_processes; ++proc_ind)
                {
                    if (proc_list[proc_ind].weight < min_weight)
                    {
                        min_proc = proc_ind;
                        min_weight = proc_list[proc_ind].weight;
                    }
                }
                curr_proc = min_proc;
                proc_list[curr_proc].set_mu_sigma(new_val, min_sigma_val);
            }
        }

        // Обновление весов процессов
        if (find_process)
        {
            for (size_t proc_ind = 0; proc_ind < created_processes; ++proc_ind)
            {
                proc_list[proc_ind].weight = (1 - alpha3) * proc_list[proc_ind].weight + alpha3 * ((proc_ind == curr_proc) ? 1 : 0);
            }
        }

        return proc_list[curr_proc].weight > weight_threshold;
#endif
    }

    MEAS_T CurrValue() const
    {
        return m_procList[m_currProc].CurrValue();
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
};
