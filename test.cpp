#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

struct LinearRegression {
    double intercept;
    double slope;

    LinearRegression() : intercept(0), slope(0) {}

    void fit(const vector<double>& X, const vector<double>& Y, double learning_rate = 0.01, int iterations = 1000) {
        int n = X.size();

        for (int i = 0; i < iterations; ++i) {
            double d_intercept = 0;
            double d_slope = 0;

            // Parallelize gradient computation
            #pragma omp parallel for reduction(+:d_intercept, d_slope)
            for (int j = 0; j < n; ++j) {
                double prediction = (slope * X[j]) + intercept;
                double error = prediction - Y[j];

                d_intercept += error;
                d_slope += error * X[j];
            }

            // Update intercept and slope
            intercept -= (learning_rate * d_intercept) / n;
            slope -= (learning_rate * d_slope) / n;
        }
    }

    double predict(double x) const {
        return slope * x + intercept;
    }

    vector<double> predict(const vector<double>& X) const {
        vector<double> predictions(X.size());
        
        #pragma omp parallel for
        for (int i = 0; i < X.size(); ++i) {
            predictions[i] = predict(X[i]);
        }
        
        return predictions;
    }
};

int main() {
    // Example data
    vector<double> X = {1, 2, 3, 4, 5};
    vector<double> Y = {2, 4, 6, 8, 10};

    // Create and train the model
    LinearRegression model;
    model.fit(X, Y);

    cout << "Intercept: " << model.intercept << endl;
    cout << "Slope: " << model.slope << endl;

    // Predict on new data
    double test_point = 12.0;
    cout << "Prediction for " << test_point << " : " << model.predict(test_point) << endl;

    return 0;
}

