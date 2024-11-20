#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

class Reader {

    private:
        static int* countLines(string path){
            ifstream file(path);

            if(!file.is_open()){
                throw string("Unable to open file for counting.");
            }

            string line;
            int rows = 1;
            
            getline(file, line);
            int cols = 0;
            for (char ch : line) {
                if (ch == ',') {
                    cols++;
                }
            }

            while(getline(file, line)){
                rows++;
            }
            
            file.close();

            int* size = new int[2];
            size[0] = rows;
            size[1] = cols;
            
            return size;
        }

    public:
        static pair<pair<int, int>, double**> readCSV(string path){

            int* size = countLines(path);
            int rows = size[0];
            int cols = size[1];

            delete size;

            double** data = new double*[rows];

            for(int i = 0; i<rows; i++){
                data[i] = new double[cols];
            }

            ifstream file(path);

            if(!file.is_open()){
                throw string("Unable to open file for reading.");
            }

            string line;
            getline(file, line);
            int i = 0;
            while(getline(file, line)){
                stringstream ss(line);

                string val;
                bool any_null_values = false;
                int j = 0;
                while(getline(ss, val, ',')){

                    if(val == ""){
                        any_null_values = true;
                        break;
                    }
                    else{
                        data[i][j] = stod(val);   
                    }
                    j++;
                }

                if(!any_null_values){
                    i++;
                }

            }

            file.close();

            return {{i, cols}, data};
        }

};

class LinearRegression {

    private:
        double* weights;
        double bias;

        int rows;
        int cols;

    public:

        LinearRegression(int rows, int cols){
            this->rows = rows;
            this->cols = cols;
            
            weights = new double[cols];
            for(int i=0; i<cols; i++){
                weights[i] = 0;
            }

            bias = 0;
        }

        double f(double x[]){
            double result = 0;
            
            for(int i=0; i<cols; i++){
                result += weights[i] * x[i];
            }
            result += bias;

            return result;
        }

        double* fit(double** data, double learning_rate, int epochs){

            double* cost_history = new double[epochs+1];

            cost_history[0] = compute_cost(data);

            // # pragma omp parallel for
            for(int i = 0; i<epochs; i++){
                auto gradients = compute_gradients(data);
                
                double* gradients_w = gradients.first;
                double gradient_b = gradients.second;

                for(int j=0; j<cols; j++){
                    cout << gradients_w[j] << " ";
                }
                cout << endl << gradient_b << endl;

                bias = bias - learning_rate * gradient_b;

                delete gradients_w;

                cost_history[i+1] = compute_cost(data);
                cout << cost_history[i+1] << endl;

            }

            return cost_history;
        }

        pair<double*, double> compute_gradients(double** data){

            double* gradients_w = new double[cols];
            for(int i = 0; i<cols; i++){
                gradients_w[i] = 0;
            }

            double gradient_b = 0;

            for(int i = 0; i<rows; i++){
                double* x = data[i];
                double y = data[i][cols];

                double difference = f(x) - y;

                for(int j=0; j<cols; j++){
                    gradients_w[j] += difference * x[j];
                }

                gradient_b += difference;
            }

            for(int i = 0; i<cols; i++){
                gradients_w[i] /= rows;
            }
            
            gradient_b /= rows;

            return {gradients_w, gradient_b};
        }

        double compute_cost(double** data){

            double loss = 0;

            for(int i = 0; i<rows; i++){
                loss += pow(f(data[i]) - data[i][cols], 2);
            }
            loss /= 2*rows;

            return loss;
        }

        double predict(double x[]){
            return f(x);
        }

        ~LinearRegression(){
            delete weights;
        }
};

int main(){

    double** data;
    int rows, cols;

    double start, end;

    try{
        auto output = Reader::readCSV("./housing.csv");
        
        rows = output.first.first;
        cols = output.first.second;

        data = output.second;
    }
    catch(string error){
        cout << error << endl;
        return -1;
    }

    LinearRegression model(rows, cols);

    start = omp_get_wtime();

    int epochs = 100;
    double* cost_history = model.fit(data, 0.5, epochs);
    
    end = omp_get_wtime();

    printf("Cost History:\n");
    for(int i=0; i<epochs+1; i++){
        printf("Epoch %d: %f\n", i, cost_history[i]);
    }

    delete cost_history;

    double prediction = model.predict(data[8]);

    printf("\nPrediction: %f\n", prediction);

    printf("Time: %f\n", end - start);

    for(int i = 0; i<cols; i++){
        delete data[i];
    }
    delete[] data;

}