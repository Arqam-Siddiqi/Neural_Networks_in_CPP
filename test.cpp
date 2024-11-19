#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

class LinearRegression {
private:
    std::vector<double> weights;
    double bias;
    double learningRate;
    int epochs;

public:
    LinearRegression(double lr, int ep) : learningRate(lr), epochs(ep) {
        bias = 0.0;
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        int n_samples = X.size();
        int n_features = X[0].size();

        // Initialize weights to zero
        weights.resize(n_features, 0.0);

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> weight_gradients(n_features, 0.0);
            double bias_gradient = 0.0;

            // Parallel computation of gradients
            // #pragma omp parallel for num_threads(4) reduction(+:bias_gradient) schedule(static)
            for (int i = 0; i < n_samples; ++i) {
                double y_pred = bias;
                for (int j = 0; j < n_features; ++j) {
                    y_pred += weights[j] * X[i][j];
                }

                double error = y_pred - y[i];

                bias_gradient += error;
                for (int j = 0; j < n_features; ++j) {
                    // #pragma omp atomic
                    weight_gradients[j] += error * X[i][j];
                }
            }

            // Update weights and bias
            bias -= (learningRate * bias_gradient) / n_samples;

            for (int j = 0; j < n_features; ++j) {
                weights[j] -= (learningRate * weight_gradients[j]) / n_samples;
            }
        }
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) {
        int n_samples = X.size();
        int n_features = X[0].size();
        std::vector<double> predictions(n_samples);

        // #pragma omp parallel for num_threads(4) schedule(static)
        for (int i = 0; i < n_samples; ++i) {
            double y_pred = bias;
            for (int j = 0; j < n_features; ++j) {
                y_pred += weights[j] * X[i][j];
            }
            predictions[i] = y_pred;
        }

        return predictions;
    }

    void printWeights() {
        std::cout << "Weights: ";
        for (double w : weights) {
            std::cout << w << " ";
        }
        std::cout << "\nBias: " << bias << std::endl;
    }
};

int main() {
    // Example dataset
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0}; // Linear relation: y = 1 + 2x_1 + x_2

    double learningRate = 0.01;
    int epochs = 1000;

    double start = omp_get_wtime();
    LinearRegression model(learningRate, epochs);
    model.fit(X, y);

    double end = omp_get_wtime();

    std::cout << end - start << std::endl;

    model.printWeights();

    // Predict
    std::vector<std::vector<double>> X_test = {
        {5.0, 6.0},
        {6.0, 7.0}
    };

    std::vector<double> predictions = model.predict(X_test);
    std::cout << "Predictions: ";
    for (double pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    return 0;
}
