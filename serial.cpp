#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

class Contents {

    public:
        int rows;
        int cols;
        int rows_in_file;

        double** data;

        Contents(int rows, int cols, int rows_in_file, double** data){
            this->rows = rows;
            this->cols = cols;
            this->rows_in_file = rows_in_file;
            this->data = data;
        }
        
};

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
        static Contents readCSV(string path){

            int* size = countLines(path);
            int rows = size[0];
            int cols = size[1];

            delete[] size;

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

            Contents obj(i, cols, rows, data);

            return obj;
        }

};

class StandardScaler {

    public:
        static void scale(double** data, int rows, int cols) {
            double* mean = new double[cols];
            calculateMean(data, rows, cols, mean);

            double* stdDev = new double[cols];
            calculateStdDev(data, rows, cols, mean, stdDev);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    data[i][j] = (data[i][j] - mean[j]) / stdDev[j];
                }
            }

            delete[] mean;
            delete[] stdDev;
        }

    private:
        
        static void calculateMean(double** data, int rows, int cols, double* mean) {
            
            for (int j = 0; j < cols; ++j) {
                mean[j] = 0.0;
                for (int i = 0; i < rows; ++i) {
                    mean[j] += data[i][j];
                }
                mean[j] /= rows;
            }
        }

        static void calculateStdDev(double** data, int rows, int cols, double* mean, double* stdDev) {
            for (int j = 0; j < cols; ++j) {
                stdDev[j] = 0.0;
                for (int i = 0; i < rows; ++i) {
                    stdDev[j] += std::pow(data[i][j] - mean[j], 2);
                }
                stdDev[j] = std::sqrt(stdDev[j] / rows);
            }
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

            for(int i = 0; i<epochs; i++){
                
                auto gradients = compute_gradients(data);
                
                double* gradients_w = gradients.first;
                double gradient_b = gradients.second;

                for(int j = 0; j<cols; j++){
                    weights[j] = weights[j] - learning_rate * gradients_w[j];
                }

                bias = bias - learning_rate * gradient_b;

                delete[] gradients_w;

                cost_history[i+1] = compute_cost(data);
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

        double getScore(double** data){
            
            // Task 1
            double mean = 0;
            for(int i = 0; i<rows; i++){
                mean += data[i][cols];
            }
            mean /= rows;
            
            // Task 2 
            double residual_sum_of_squares = compute_cost(data) * 2 * rows;

            double total_sum_of_squares = 0;
            for(int i = 0; i<rows; i++){
                double diff = data[i][cols] - mean;
                total_sum_of_squares += pow(diff, 2);
            }

            double r2 = 1 - residual_sum_of_squares / total_sum_of_squares;

            return r2;
        }

        ~LinearRegression(){
            delete[] weights;
        }
};



int main(){

    double** data;
    int rows, cols, rows_in_file;
    
    double start, end;
    start = omp_get_wtime();

    try{
        double temp1 = omp_get_wtime();
        auto output = Reader::readCSV("./samples/sensor.csv");
        double temp2 = omp_get_wtime();
        cout << temp2 - temp1 << endl;

        rows = output.rows;
        cols = output.cols;
        rows_in_file = output.rows_in_file;

        data = output.data;
    }
    catch(string error){
        cout << error << endl;
        return -1;
    }

    double temp1 = omp_get_wtime();
    StandardScaler::scale(data, rows, cols);
    double temp2 = omp_get_wtime();
    cout << temp2 - temp1 << endl;

    LinearRegression model(rows, cols);


    int epochs = 100;
    double alpha = 0.05;
    double* cost_history = model.fit(data, alpha, epochs);

    cout << "Final Cost: " << cost_history[epochs] << endl;
    cout << "Improvement in Cost: " << cost_history[0] / cost_history[epochs] * 100 << "%" << endl;

    delete[] cost_history;
    
    
    double prediction = model.predict(data[9]);
    cout << "Prediction: " << prediction << endl;
    
    double score = model.getScore(data);
    cout << "R2: " << score << endl;

    end = omp_get_wtime();
    cout << "\nTime Taken: " << end - start << endl;

    delete[] data;
    
}