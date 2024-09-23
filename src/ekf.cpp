#include "ekf.hpp"

EKF::EKF(int state_dim, int meas_dim, const Eigen::VectorXf &detection)
    : state_dim_(state_dim), meas_dim_(meas_dim),
      state_(detection),
      covariance_(Eigen::MatrixXf::Identity(state_dim, state_dim)),
      process_noise_(Eigen::MatrixXf::Identity(state_dim, state_dim)),
      measurement_noise_(Eigen::MatrixXf::Identity(meas_dim, meas_dim)),
      R(Eigen::MatrixXf::Identity(meas_dim, meas_dim) * 0.1f),
      d_state_(nullptr), d_covariance_(nullptr){
    allocateDeviceMemory();
    copyToDevice();
}

EKF::~EKF() {
    freeDeviceMemory();
}

void EKF::predict(const Eigen::VectorXf &control) {
    // // Copy control input to device (if needed)
    float *d_control;
    cudaMalloc((void**)&d_control, control.size() * sizeof(float));
    cudaMemcpy(d_control, control.data(), control.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel for prediction
    launchPredictKernel();

    // Copy results back to host
    copyToHost();

    cudaFree(d_control);
}

void EKF::update(const Eigen::VectorXf &measurement) {

    Eigen::MatrixXf F = jacobianState(state_, Eigen::VectorXf::Zero(state_dim_));
    Eigen::MatrixXf H = jacobianMeasurement(state_);
    
    // Allocate and copy matrices to device
    float *d_measurement;
    float *d_H;
    float *d_F;
    float *d_R;

    cudaMalloc((void**)&d_measurement, meas_dim_ * sizeof(float));
    cudaMalloc((void**)&d_H, meas_dim_ * state_dim_ * sizeof(float));
    cudaMalloc((void**)&d_F, state_dim_ * state_dim_ * sizeof(float));
    cudaMalloc((void**)&d_R, meas_dim_ * meas_dim_ * sizeof(float));

    cudaMemcpy(d_measurement, measurement.data(), meas_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H, H.data(), meas_dim_ * state_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F.data(), state_dim_ * state_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R.data(), meas_dim_ * meas_dim_ * sizeof(float), cudaMemcpyHostToDevice);

    // // Launch kernel
    float dt = 1.0f / 30;
    launchUpdateKernel(d_measurement, d_H, d_F, d_R, state_dim_, meas_dim_, dt);
    copyToHost();

    // // Free device memory
    cudaFree(d_measurement);
    cudaFree(d_H);
    cudaFree(d_F);
    cudaFree(d_R);
}

Eigen::VectorXf EKF::getState() const {
    return state_;
}

Eigen::MatrixXf EKF::jacobianState(const Eigen::VectorXf &state, const Eigen::VectorXf &control) {
    Eigen::MatrixXf F(state_dim_, state_dim_);
    float dt = 1.0; // Time step (this can be parameterized)

    F.setIdentity();
    F(0, 2) = dt; // ∂x/∂vx
    F(1, 3) = dt; // ∂y/∂vy

    return F;
}

Eigen::MatrixXf EKF::jacobianMeasurement(const Eigen::VectorXf &state) {
    Eigen::MatrixXf H(meas_dim_, state_dim_);

    H.setZero();
    H(0, 0) = 1.0; // ∂z_x/∂x
    H(1, 1) = 1.0; // ∂z_y/∂y

    return H;
}

void EKF::allocateDeviceMemory() {
    cudaMalloc((void**)&d_state_, state_dim_ * sizeof(float));
    cudaMalloc((void**)&d_covariance_, state_dim_ * state_dim_ * sizeof(float));
    cudaDeviceSynchronize();
}

void EKF::freeDeviceMemory() {
    cudaFree(d_state_);
    cudaFree(d_covariance_);
}

void EKF::copyToDevice() {
    cudaMemcpy(d_state_, state_.data(), state_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_covariance_, covariance_.data(), state_dim_ * state_dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

void EKF::copyToHost() {
    cudaMemcpy(state_.data(), d_state_, state_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(covariance_.data(), d_covariance_, state_dim_ * state_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
}
