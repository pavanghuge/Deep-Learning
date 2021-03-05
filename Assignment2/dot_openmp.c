#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


// Function declaration for Kernel

void kernel(unsigned int, unsigned int,float*, float*, float*, unsigned int );

// the main file

int main(int argc, char* argv[]){
 
 
 printf("Program Started : ");

    // Input data file 
    FILE *data;
    data = fopen(argv[1],"r");
    if(data == NULL){
        printf("\n The Data file cannot be opened!!!");
        return 0;
    }
    // Input weights file
    FILE *weight;

    weight = fopen(argv[2],"r");
    if(weight == NULL){

        printf("\n The Weights file cannot be opened!!!");
        return 0;
    }
    // Other input parameters rows, columns, and number of processors
    unsigned int rows=atoi(argv[3]);
    unsigned int cols=atoi(argv[4]);
    int nprocs = atoi(argv[5]);
    


    float* result = (float*) malloc(cols * sizeof(float));
    

    // Allocating memory of input file and vector file
    
    size_t data_size = 0;
    data_size = (size_t)((size_t)rows * (size_t)cols);
    size_t weight_size = 0;
    int weight_row =1;
    weight_size = (size_t)((size_t)weight_row * (size_t)cols);

    printf("\nNumber of elements in matrix file= %lu",data_size);
    printf("\nNumber of elements in weights file = %lu",weight_size);

    fflush(stdout);

    float *dataT = (float*)malloc((data_size)*sizeof(float));
    float *dataV = (float*)malloc((weight_size)*sizeof(float));

    if(dataT == NULL) {
            printf("\n ERROR: Memory for data not allocated.\n");
    }
    if(dataV == NULL) {
            printf("\n ERROR: Memory for data not allocated.\n");
    }

    float mat[rows][cols];
    float variable;
    unsigned long i;
    unsigned long j;

     // Transfer the Data from the file to CPU Memory 
    for (i=0;i<rows;i++){
            for (j=0;j<cols;j++){
                    fscanf(data,"%f",&variable);
                    mat[i][j]=variable;
            }
    }
    for (i=0;i<cols;i++){
            for (j=0;j<rows;j++){
                    dataT[rows*i+j]=mat[j][i];
                    }
    }

    for (j=0;j<cols;j++){
            fscanf(weight,"%f",&dataV[j]);
    }

    fclose(data);
    fclose(weight);
    printf("\nData reading complete\n");

    unsigned int jobs;
    float* multiplication = (float*) malloc(rows * sizeof(float));
    struct timeval starttime, endtime;
    float seconds;
    


    jobs = (unsigned int) ((rows+nprocs-1)/nprocs);

     gettimeofday(&starttime, NULL);
    /*Calling the kernel function */

    printf("\nNumber jobs = %d\n", jobs);

#pragma omp parallel num_threads(nprocs)
    kernel(rows,cols,dataT,dataV,multiplication,jobs);
    gettimeofday(&endtime, NULL); 
    seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
    printf("\nTime taken by kernel function = %f\n", seconds);
    printf("\n******** Output ***********\n\n"); 
    for(i = 0; i < cols; i++) {
        printf("%f ", multiplication[i]);
        printf("\n");
    }
    printf("\n");


    printf("Program Ended :");

    return 0;

}// Main Closes



void kernel(unsigned int rows, unsigned int cols ,float *data, float *weight,float *results, unsigned int jobs){
        int i,j,stop;
        float dot_product = 0;

	int tid = omp_get_thread_num();
	    if((tid+1)*jobs > rows) 
            stop=rows;
        else 
            stop = (tid+1)*jobs;

        for (j = tid*jobs; j < stop; j++) { 
		dot_product = 0;
	        for ( i = 0 ; i < cols ; i++ ) {
	               dot_product += data[i*rows+j] * weight[i];
	        }

        
        	results[j] = dot_product;
        }
}