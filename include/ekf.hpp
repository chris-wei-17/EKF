#ifndef EKF_HPP
#define EKF_HPP

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>

class EKF {
public:
    EKF(int state_dim, int meas_dim, const Eigen::VectorXf &detection);
    ~EKF();
    void predict(const Eigen::VectorXf& control);
    void update(const Eigen::VectorXf& measurement);

    Eigen::VectorXf getState() const;

    int missedDetectionCount = 0;
    Eigen::VectorXf state_;

private:
    int state_dim_;
    int meas_dim_;
    Eigen::MatrixXf covariance_;
    Eigen::MatrixXf process_noise_;
    Eigen::MatrixXf measurement_noise_;
    Eigen::MatrixXf R;
    // CUDA
    void launchPredictKernel();
    void launchUpdateKernel(float *d_measurement, float *d_H, float *d_F, float *d_R, int state_dim, int meas_dim, float dt);

    float* d_state_;
    float* d_covariance_;

    Eigen::VectorXf stateTransitionFunction(const Eigen::VectorXf& state, const Eigen::VectorXf& control);
    Eigen::VectorXf measurementFunction(const Eigen::VectorXf& state);

    Eigen::MatrixXf jacobianState(const Eigen::VectorXf& state, const Eigen::VectorXf& control);
    Eigen::MatrixXf jacobianMeasurement(const Eigen::VectorXf& state);

    void allocateDeviceMemory();
    void freeDeviceMemory();
    void copyToDevice();
    void copyToHost();
};

#endif // EKF_HPP
