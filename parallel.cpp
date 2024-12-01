#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
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

            # pragma omp simd
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
        void scale(double** data, int rows, int cols) {
            
            double mean[cols];
            calculateMean(data, rows, cols, mean);

            double stdDev[cols];
            calculateStdDev(data, rows, cols, mean, stdDev);

            // for(int i = 0; i<cols; i++){
            //     cout << i << " " << stdDev[i] << " " << mean[i] << endl;
            // }
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if(stdDev[j] != 0)
                        data[i][j] = (data[i][j] - mean[j]) / stdDev[j];

                }
            }

        }

    private:
        
        void calculateMean(double** data, int rows, int cols, double* mean) {
            
            for (int j = 0; j < cols; j++) {
                double sum = 0.0;
            
                #pragma omp simd reduction(+:sum)
                for (int i = 0; i < rows; i++) {
                    sum += data[i][j];
                }
                
                mean[j] = sum / rows;
            }
            
        }

        void calculateStdDev(double** data, int rows, int cols, double* mean, double* stdDev) {
            for (int j = 0; j < cols; ++j) {
                double variance = 0.0;

                #pragma omp simd reduction(+:variance)
                for (int i = 0; i < rows; ++i) {
                    variance += std::pow(data[i][j] - mean[j], 2);
                }

                stdDev[j] = std::sqrt(variance / rows);
            }
        }

};

class Regression {

    public:
        virtual double* fit(double** data, double learning_rate, int epochs) = 0;
        virtual double getScore(double** data) = 0;
        virtual double f(double x[]) = 0;
        virtual pair<double*, double> compute_gradients(double** data) = 0;
        virtual double compute_cost(double** data) = 0;
        virtual double predict(double x[]) = 0;

};

class LinearRegression : public Regression{

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

        double f(double x[]) {
            double result = 0;

            #pragma omp simd reduction(+:result)
            for (int i = 0; i < cols; i++) {
                result += weights[i] * x[i];
            }
            result += bias;

            return result;
        }

        double* fit(double** data, double learning_rate, int epochs){

            double* cost_history = new double[epochs+1];

            cost_history[0] = compute_cost(data);

            // Uncomment the next line for a massive increase in performance in exchange for a very slight dip in score
            // # pragma omp parallel for
            for(int i = 0; i<epochs; i++){
                
                auto gradients = compute_gradients(data);
                
                double* gradients_w = gradients.first;
                double gradient_b = gradients.second;

                # pragma omp simd
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

            # pragma omp simd
            for(int i = 0; i<cols; i++){
                gradients_w[i] = 0;
            }

            double gradient_b = 0;

            for(int i = 0; i<rows; i++){
                double* x = data[i];
                double y = data[i][cols];

                double difference = f(x) - y;
                
                # pragma omp simd
                for(int j=0; j<cols; j++){
                    gradients_w[j] += difference * x[j];
                }
    
                gradient_b += difference;
            }

            # pragma omp simd
            for(int i = 0; i<cols; i++){
                gradients_w[i] /= rows;
            }
            
            gradient_b /= rows;

            return {gradients_w, gradient_b};
        }

        double compute_cost(double** data){

            double loss = 0;

            # pragma omp parallel for reduction(+:loss)
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
            
            double mean = 0;
            double residual_sum_of_squares = 0;

            #pragma omp parallel
            {
                #pragma omp single
                {
                    // Task 1: Calculate mean
                    #pragma omp task shared(mean)
                    {
                        double local_mean = 0;
                        #pragma omp parallel for reduction(+: local_mean)
                        for (int i = 0; i < rows; i++) {
                            local_mean += data[i][cols];
                        }
                        mean = local_mean / rows;
                    }

                    // Task 2: Calculate residual sum of squares
                    #pragma omp task shared(residual_sum_of_squares)
                    {
                        residual_sum_of_squares = compute_cost(data) * 2 * rows;
                    }
                }
            }


            double total_sum_of_squares = 0;

            # pragma omp parallel for reduction(+: total_sum_of_squares)
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

class LogisticRegression : public Regression {

    private:
        double* weights;
        double bias;

        int rows;
        int cols;

    public:

        LogisticRegression(int rows, int cols){
            this->rows = rows;
            this->cols = cols;
            
            weights = new double[cols];
            for(int i=0; i<cols; i++){
                weights[i] = 0;
            }

            bias = 0;
        }

        double f(double x[]) {
            double result = 0;

            #pragma omp simd reduction(+:result)
            for (int i = 0; i < cols; i++) {
                result += weights[i] * x[i];
            }
        
            result += bias;
            
            result = 1 / (1 + exp(-result));

            return result;
        }

        double* fit(double** data, double learning_rate, int epochs){

            double* cost_history = new double[epochs+1];

            cost_history[0] = compute_cost(data);

            // Uncomment the next line for a massive increase in performance in exchange for a very slight dip in score
            // # pragma omp parallel for
            for(int i = 0; i<epochs; i++){
                
                auto gradients = compute_gradients(data);
                
                double* gradients_w = gradients.first;
                double gradient_b = gradients.second;

                # pragma omp simd
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

            # pragma omp simd
            for(int i = 0; i<cols; i++){
                gradients_w[i] = 0;
            }

            double gradient_b = 0;

            for(int i = 0; i<rows; i++){
                double* x = data[i];
                double y = data[i][cols];

                double difference = f(x) - y;
                
                # pragma omp simd
                for(int j=0; j<cols; j++){
                    gradients_w[j] += difference * x[j];
                }
    
                gradient_b += difference;
            }

            # pragma omp simd
            for(int i = 0; i<cols; i++){
                gradients_w[i] /= rows;
            }
            
            gradient_b /= rows;

            return {gradients_w, gradient_b};
        }

        double compute_cost(double** data){

            double loss = 0;

            # pragma omp parallel for reduction(+:loss)
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
            
            int correct_predictions = 0;

            #pragma omp parallel for reduction(+:correct_predictions)
            for (int i = 0; i < rows; i++) {
                double* x = data[i];
                double actual = data[i][cols];

                double predicted_prob = f(x);
                double predicted;
                
                if(predicted_prob >= 0.5){
                    predicted = 1;
                }
                else{
                    predicted = 0;
                }

                if (predicted == actual) {
                    correct_predictions++;
                }
            }

            return static_cast<double>(correct_predictions) / rows;

        }

        ~LogisticRegression(){
            delete[] weights;
        }

};


int main(){

    double** data;
    int rows, cols, rows_in_file;
    
    double start, end;
    start = omp_get_wtime();

    try{
        // auto output = Reader::readCSV("./samples/sensor.csv");
        auto output = Reader::readCSV("./samples/network_intrusion.csv");
        
        rows = output.rows;
        cols = output.cols;
        rows_in_file = output.rows_in_file;

        data = output.data;
    }
    catch(string error){
        cout << error << endl;
        return -1;
    }

    StandardScaler scaler;
    scaler.scale(data, rows, cols);

    LogisticRegression model(rows, cols);

    int epochs = 100;
    double alpha = 0.05;
    double* cost_history = model.fit(data, alpha, epochs);
    
    cout << "Final Cost: " << cost_history[epochs] << endl;
    cout << "Improvement in Cost: " << cost_history[0] / cost_history[epochs] * 100 << "%" << endl;

    delete[] cost_history;
        
    // double prediction = model.predict(data[9]);
    // cout << "Prediction: " << prediction << endl;

    double score = model.getScore(data);
    cout << "Score: " << score << endl;

    end = omp_get_wtime();
    cout << "\nTime Taken: " << end - start << endl;

    delete[] data;
    
}