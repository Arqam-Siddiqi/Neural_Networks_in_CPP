#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <vector>

using namespace std;

class LinearRegression {

    private:
        double* weights;
        double bias;
        int n;

    public:

        LinearRegression(int n){
            this->n = n;
            
            weights = new double[n];
            for(int i=0; i<n; i++){
                weights[i] = 0;
            }

            bias = 0;
        }

        double f(double x){
            return weights[0] * x + bias;
        }

        double* fit(double** data, int length, double learning_rate, int epochs){

            double* cost_history = new double[epochs+1];

            cost_history[0] = compute_cost(data, length);

            # pragma omp parallel for
            for(int i = 0; i<epochs; i++){
                double* gradients = compute_gradients(data, length);

                double temp_w[n];
                for(int i=0; i<n; i++){
                    temp_w[i] = weights[i] - learning_rate * gradients[0];
                }
                double temp_b = bias - learning_rate * gradients[1];

                delete gradients;

                for(int i=0; i<n; i++){
                    weights[i] = temp_w[i];
                }
                bias = temp_b;

                cost_history[i+1] = compute_cost(data, length);
            }

            return cost_history;
        }

        double* compute_gradients(double** data, int length){

            double* gradients = new double[2];

            for(int i = 0; i<length; i++){
                double x = data[i][0], y = data[i][1];
                gradients[0] = (f(x) - y) * x;
                gradients[1] = (f(x) - y);
            }

            gradients[0] /= length;
            gradients[1] /= length;

            return gradients;
        }

        double compute_cost(double** data, int length){

            double loss = 0;

            for(int i = 0; i<length; i++){
                loss = pow(f(data[i][0]) - data[i][1], 2);
            }
            loss /= length;

            return loss;
        }

        double predict(double x){
            return weights[0] * x + bias;
        }

};

class Reader {

    private:
        int* countLines(string path){
            ifstream file(path);

            if(!file.is_open()){
                throw string("Unable to open file for counting.");
            }

            string line;
            int rows = 0;

            while(getline(file, line)){
                rows++;
            }
            
            file.close();

            int cols = 0;
            for (char ch : line) {
                if (ch == ',') {
                    ++cols;
                }
            }

            int* size = new int[2];
            size[0] = rows;
            size[1] = cols;
            
            return size;
        }

    public:
        double** readCSV(string path){


                int* size = countLines(path);
                int rows = size[0];
                int cols = size[1];

                delete size;

                double** data = new double*[rows];

                for(int i = 0; i<cols; i++){
                    data[i] = new double[cols];
                }
                

                ifstream file(path);

                if(!file.is_open()){
                    throw string("Unable to open file for reading.");
                }

                int sum = 0;
                // #pragma omp parallel for num_threads(4)
                for(int i = 0; i<rows; i++){
                    sum = i*3;
                }

                cout << sum << endl;

                file.close();

            
        }

};

int main(){

    double** data;

    double start = omp_get_wtime();

    try{
        Reader r1;
        data = r1.readCSV("./housing.csv");
    }
    catch(string error){
        cout << error << endl;
        return -1;
    }

    double end = omp_get_wtime();

    cout << end - start << endl;

    return 0;
    

    int length = sizeof(data)/sizeof(data[0]);

    LinearRegression model(1);

    start = omp_get_wtime();

    int epochs = 100;
    double* cost_history = model.fit(data, length, 0.01, epochs);
    
    end = omp_get_wtime();

    printf("Cost History:\n");
    for(int i=0; i<epochs+1; i++){
        printf("Epoch %d: %.14f\n", i, cost_history[i]);
    }

    delete cost_history;

    double prediction = model.predict(11);

    printf("\nPrediction: %.14f\n", prediction);

    printf("Time: %f\n", end - start);

}