#ifndef CSV_TO_EIGEN_HPP
#define CSV_TO_EIGEN_HPP

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace csvutil
{

    inline Eigen::MatrixXd readCSVtoEigen(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open CSV file: " + filename);
        }

        std::vector<std::vector<double>> data;
        std::string line, cell;

        while (std::getline(file, line))
        {
            std::stringstream linestream(line);
            std::vector<double> row;

            while (std::getline(linestream, cell, ','))
            {
                try
                {
                    row.push_back(std::stod(cell));
                }
                catch (const std::invalid_argument &e)
                {
                    throw std::runtime_error("Invalid number in CSV: '" + cell + "'");
                }
            }

            if (!row.empty())
            {
                data.push_back(row);
            }
        }

        file.close();

        if (data.empty())
        {
            throw std::runtime_error("CSV file is empty or only contains headers");
        }

        const std::size_t rows = data.size();
        const std::size_t cols = data[0].size();
        Eigen::MatrixXd mat(rows, cols);

        for (std::size_t i = 0; i < rows; ++i)
        {
            if (data[i].size() != cols)
            {
                throw std::runtime_error("Inconsistent number of columns at row " + std::to_string(i));
            }
            for (std::size_t j = 0; j < cols; ++j)
            {
                mat(i, j) = data[i][j];
            }
        }

        return mat;
    }

    inline void saveEigenToCSV(const std::string &filename, const Eigen::MatrixXd &matrix)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        const Eigen::Index rows = matrix.rows();
        const Eigen::Index cols = matrix.cols();

        for (Eigen::Index i = 0; i < rows; ++i)
        {
            for (Eigen::Index j = 0; j < cols; ++j)
            {
                file << matrix(i, j);
                if (j < cols - 1)
                    file << ",";
            }
            file << "\n";
        }

        file.close();
    }

    inline void logVectorToCSV(const Eigen::VectorXd &vec, const std::string &filename)
    {
        std::ofstream file(filename, std::ios_base::app); // append mode
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        for (int i = 0; i < vec.size(); ++i)
        {
            file << vec[i];
            if (i < vec.size() - 1)
                file << ",";
        }
        file << "\n";
    }

    inline void saveVectorListToCSV(const std::string &filename,
                                    const std::vector<Eigen::VectorXd> &rows)
    {
        // Open with truncation: clears existing file or creates a new one.
        std::ofstream file(filename, std::ios::out | std::ios::trunc);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        if (rows.empty())
        {
            // Nothing to write; leave an empty file.
            return;
        }

        const Eigen::Index cols = rows.front().size();
        // Ensure all rows have the same length
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            if (rows[i].size() != cols)
            {
                throw std::runtime_error("Row " + std::to_string(i) +
                                         " has size " + std::to_string(rows[i].size()) +
                                         " but expected " + std::to_string(cols));
            }
        }

        // Write each VectorXd as a CSV row
        for (const auto &v : rows)
        {
            for (Eigen::Index j = 0; j < cols; ++j)
            {
                file << v[j];
                if (j < cols - 1)
                    file << ",";
            }
            file << "\n";
        }
        file.close();
    }
} // namespace csvutil

#endif // CSV_TO_EIGEN_HPP
