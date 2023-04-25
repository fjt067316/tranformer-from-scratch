#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>

using namespace std;

#pragma once
// https://www.youtube.com/watch?v=1JmZ5idFcVI 
// Layer norm in transformers might be done row by row and not by entire layer ????
class LayerNorm {
public:
    double gamma = 1;
    double beta = 0;

    LayerNorm() {}

    vector<vector<double>> forward(vector<vector<double>> input) {
        // Compute mean and variance of the input
        double mean = 0.0;
        double variance = 0.0;
        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                mean += input[i][j];
            }
        }
        mean /= input.size()*input[0].size();

        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                variance += pow((input[i][j]-mean), 2);
            }
        }

        variance /= input.size()*input[0].size();

        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                input[i][j] = (input[i][j]-mean)/(variance+1e-9);
                input[i][j] *= gamma;
                input[i][j] += beta;
            }
        }
        return input;
    }

    // void backward(vector<float> grad_output, vector<float>& grad_input) {
    //     // Compute gradients with respect to gamma and beta
    //     std::vector<float> grad_gamma(num_features_);
    //     std::vector<float> grad_beta(num_features_);
    //     for (int i = 0; i < num_features_; i++) {
    //         grad_gamma[i] = grad_output[i] * input_norm_[i];
    //         grad_beta[i] = grad_output[i];
    //     }

    //     // Compute gradient of normalized input
    //     std::vector<float> grad_input_norm(num_features_);
    //     for (int i = 0; i < num_features_; i++) {
    //         grad_input_norm[i] = grad_output[i] * gamma_[i];
    //     }

    //     // Compute gradients of mean and variance
    //     float grad_mean = 0.0f;
    //     float grad_variance = 0.0f;
    //     for (int i = 0; i < num_features_; i++) {
    //         grad_mean += grad_input_norm[i];
    //         grad_variance += grad_input_norm[i] * (input_[i] - mean_) * -0.5f * std::pow(variance_ + eps_, -1.5f);
    //     }

    //     // Compute gradient of input
    //     grad_input.resize(num_features_);
    //     for (int i = 0; i < num_features_; i++) {
    //         grad_input[i] = grad_input_norm[i] * 1.0f / std::sqrt(variance_ + eps_) + grad_variance * 2.0f * (input_[i] - mean_) / num_features_ + grad_mean / num_features_;
    //     }

    //     // Update gamma and beta parameters
    //     for (int i = 0; i < num_features_; i++) {
    //         gamma_[i] -= lr_ * grad_gamma[i];
    //         beta_[i] -= lr_ * grad_beta[i];
    //     }
    // }

};
