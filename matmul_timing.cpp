#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <iomanip> // For std::fixed, std::setprecision

// Type alias for matrix
using Matrix = std::vector<std::vector<double>>;

// Enum for sketch types
enum class SketchType
{
    GAUSSIAN,
    RADEMACHER
};

// --- Helper Functions ---

// Function to create an empty matrix
Matrix create_matrix(size_t rows, size_t cols)
{
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

// Function to fill a matrix with random values ~ U[-1, 1]
void fill_random(Matrix &mat, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &row : mat)
    {
        for (auto &val : row)
        {
            val = dist(rng);
        }
    }
}

// Function to print a matrix (for debugging)
void print_matrix(const Matrix &mat, const std::string &name = "")
{
    if (!name.empty())
    {
        std::cout << name << " (" << mat.size() << "x" << (mat.empty() ? 0 : mat[0].size()) << "):\n";
    }
    if (mat.empty())
    {
        std::cout << "[Empty Matrix]\n";
        return;
    }
    size_t max_rows = 5;
    size_t max_cols = 5;
    for (size_t i = 0; i < std::min(mat.size(), max_rows); ++i)
    {
        std::cout << "[ ";
        for (size_t j = 0; j < std::min(mat[0].size(), max_cols); ++j)
        {
            std::cout << std::fixed << std::setprecision(3) << mat[i][j] << " ";
        }
        if (mat[0].size() > max_cols)
            std::cout << "... ";
        std::cout << "]\n";
    }
    if (mat.size() > max_rows)
        std::cout << "...\n";
    std::cout << std::endl;
}

// --- Matrix Operations ---

// Exact Matrix Multiplication C = A * B
Matrix matmul_exact(const Matrix &A, const Matrix &B)
{
    if (A.empty() || B.empty() || A[0].size() != B.size())
    {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    size_t n = A.size();
    size_t m = B.size(); // == A[0].size()
    size_t p = B[0].size();

    Matrix C = create_matrix(n, p);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < p; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

// Transpose Matrix
Matrix transpose(const Matrix &A)
{
    if (A.empty())
        return Matrix();
    size_t n = A.size();
    size_t m = A[0].size();
    Matrix T = create_matrix(m, n);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

// --- Randomized Matrix Multiplication ---

// Generate Sketch Matrix S (k x m)
Matrix generate_sketch_matrix(size_t k, size_t m, SketchType type, std::mt19937 &rng)
{
    Matrix S = create_matrix(k, m);
    double scale = 1.0 / std::sqrt(static_cast<double>(k));

    if (type == SketchType::GAUSSIAN)
    {
        // Normal distribution N(0, 1/k) -> N(0, 1) * sqrt(1/k)
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < k; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                S[i][j] = dist(rng) * scale;
            }
        }
    }
    else
    {                                                  // RADEMACHER
        std::uniform_int_distribution<int> dist(0, 1); // 0 or 1
        for (size_t i = 0; i < k; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                S[i][j] = (dist(rng) == 0 ? -1.0 : 1.0) * scale;
            }
        }
    }
    return S;
}

// Randomized Matrix Multiplication using Projection Sketch C_approx = (A * S^T) * (S * B)
Matrix matmul_randomized_projection(const Matrix &A, const Matrix &B, size_t k, SketchType type, std::mt19937 &rng)
{
    if (A.empty() || B.empty() || A[0].size() != B.size())
    {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
    }
    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();

    if (k == 0)
    {
        throw std::invalid_argument("Sketch dimension k must be positive");
    }
    // If k >= m, sketching doesn't reduce complexity in this formulation.
    // You could fall back to exact or cap k, but here we allow it for comparison.
    if (k > m)
    {
        std::cerr << "Warning: Sketch dimension k=" << k << " > inner dimension m=" << m
                  << ". Sketching may not provide speedup." << std::endl;
    }

    // 1. Generate Sketch Matrix S (k x m)
    Matrix S = generate_sketch_matrix(k, m, type, rng);

    // 2. Compute S_T (m x k)
    Matrix S_T = transpose(S);

    // 3. Compute intermediate products
    // A_ST = A * S^T  (n x m) * (m x k) -> (n x k)
    Matrix A_ST = matmul_exact(A, S_T);

    // S_B = S * B    (k x m) * (m x p) -> (k x p)
    Matrix S_B = matmul_exact(S, B);

    // 4. Compute final approximation
    // C_approx = A_ST * S_B  (n x k) * (k x p) -> (n x p)
    Matrix C_approx = matmul_exact(A_ST, S_B);

    return C_approx;
}

// --- Error Calculation ---

// Calculate Frobenius Norm
double frobenius_norm(const Matrix &A)
{
    double sum_sq = 0.0;
    for (const auto &row : A)
    {
        for (double val : row)
        {
            sum_sq += val * val;
        }
    }
    return std::sqrt(sum_sq);
}

// Calculate Relative Frobenius Norm Error: ||A - B||_F / ||A||_F
double relative_frobenius_error(const Matrix &A_exact, const Matrix &B_approx)
{
    if (A_exact.empty() || A_exact.size() != B_approx.size() || A_exact[0].size() != B_approx[0].size())
    {
        throw std::invalid_argument("Matrices must have the same dimensions for error calculation");
    }

    size_t n = A_exact.size();
    size_t p = A_exact[0].size();
    Matrix diff = create_matrix(n, p);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < p; ++j)
        {
            diff[i][j] = A_exact[i][j] - B_approx[i][j];
        }
    }

    double norm_diff = frobenius_norm(diff);
    double norm_exact = frobenius_norm(A_exact);

    if (norm_exact < 1e-10)
    { // Avoid division by zero or near-zero
        if (norm_diff < 1e-10)
            return 0.0; // Both are zero
        else
            return std::numeric_limits<double>::infinity(); // Exact is zero, approx is not
    }

    return norm_diff / norm_exact;
}

double get_condition_number(const Matrix &A)
{
    if (A.empty())
        return 0.0;
    const size_t n = A.size();
    const size_t m = A[0].size();
    std::vector<double> singular_values(std::min(n, m), 0.0);
    for (size_t i = 0; i < singular_values.size(); ++i)
    {
        singular_values[i] = frobenius_norm(A);
    }
    std::sort(singular_values.begin(), singular_values.end(), std::greater<double>());
    double cond_num = singular_values[0] / singular_values.back();
    return cond_num;
}

// --- Benchmarking ---
int main()
{
    // Setup Random Number Generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // --- Configuration ---
    size_t n = 256;                                           // Rows of A
    size_t m = 512;                                           // Cols of A / Rows of B
    size_t p = 1024;                                           // Cols of B
    std::vector<size_t> sketch_dims = {16, 32, 64, 128, 256}; // Values of k to test

    std::cout << "Benchmarking Matrix Multiplication\n";
    std::cout << "Matrix A: " << n << "x" << m << "\n";
    std::cout << "Matrix B: " << m << "x" << p << "\n";
    std::cout << "--------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Method"
              << std::setw(10) << "k"
              << std::setw(15) << "Time (ms)"
              << std::setw(20) << "Rel Frob Error" 
              << std::setw(20) << "Cond Num" 
              << std::endl;
    std::cout << "--------------------------------------------------------------------\n";

    // --- Create and Fill Matrices ---
    Matrix A = create_matrix(n, m);
    Matrix B = create_matrix(m, p);
    fill_random(A, rng);
    fill_random(B, rng);

    // --- Time Exact MatMul ---
    auto start_time = std::chrono::high_resolution_clock::now();
    Matrix C_exact = matmul_exact(A, B);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << std::left << std::setw(15) << "Exact"
              << std::setw(10) << "-"
              << std::setw(15) << duration_ms
              << std::setw(20) << "0.0" 
              << std::setw(20) << get_condition_number(C_exact)
              << std::endl;

    // --- Time Randomized MatMul ---
    for (size_t k : sketch_dims)
    {
        if (k > m)
            continue; // Optional: skip k > m if desired

        // Gaussian Sketch
        start_time = std::chrono::high_resolution_clock::now();
        Matrix C_gaussian = matmul_randomized_projection(A, B, k, SketchType::GAUSSIAN, rng);
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_gaussian = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double error_gaussian = relative_frobenius_error(C_exact, C_gaussian);

        std::cout << std::left << std::setw(15) << "Gaussian"
                  << std::setw(10) << k
                  << std::setw(15) << duration_gaussian
                  << std::fixed << std::setprecision(5) << std::setw(20) << error_gaussian 
                  << std::setw(20) << get_condition_number(C_gaussian) 
                  << std::endl;

        // Rademacher Sketch
        start_time = std::chrono::high_resolution_clock::now();
        Matrix C_rademacher = matmul_randomized_projection(A, B, k, SketchType::RADEMACHER, rng);
        end_time = std::chrono::high_resolution_clock::now();
        auto duration_rademacher = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double error_rademacher = relative_frobenius_error(C_exact, C_rademacher);

        std::cout << std::left << std::setw(15) << "Rademacher"
                  << std::setw(10) << k
                  << std::setw(15) << duration_rademacher
                  << std::fixed << std::setprecision(5) << std::setw(20) << error_rademacher 
                  << std::setw(20) << get_condition_number(C_rademacher) 
                  << std::endl;
    }
    std::cout << "--------------------------------------------------------------------\n";

    std::cout << "Condition Number of A: " << get_condition_number(A) << std::endl;
    std::cout << "Condition Number of B: " << get_condition_number(B) << std::endl;
    std::cout << "--------------------------------------------------------------------\n";

    // --- Example with smaller dimensions for verification (optional) ---
    /*
    std::cout << "\nVerification with small matrices (3x4 * 4x2):\n";
    Matrix A_small = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 1, 2, 3}};
    Matrix B_small = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    size_t k_small = 2;

    std::cout << "A_small:\n";
    print_matrix(A_small);
    std::cout << "B_small:\n";
    print_matrix(B_small);

    Matrix C_exact_small = matmul_exact(A_small, B_small);
    std::cout << "Exact Result:\n";
    print_matrix(C_exact_small);
    std::cout << std::endl;

    Matrix C_gauss_small = matmul_randomized_projection(A_small, B_small, k_small, SketchType::GAUSSIAN, rng);
    std::cout << "Gaussian Approx (k=" << k_small << "):\n";
    print_matrix(C_gauss_small);
    std::cout << "Error: " << relative_frobenius_error(C_exact_small, C_gauss_small) << std::endl
              << std::endl
              << std::endl;

    Matrix C_rade_small = matmul_randomized_projection(A_small, B_small, k_small, SketchType::RADEMACHER, rng);
    std::cout << "Rademacher Approx (k=" << k_small << "):\n";
    print_matrix(C_rade_small);
    std::cout << "Error: " << relative_frobenius_error(C_exact_small, C_rade_small) << std::endl
              << std::endl;
    */

    return 0;
}

// To run: `g++ matmul_timing.cpp -o matmul_timing -std=c++11 -O2`
