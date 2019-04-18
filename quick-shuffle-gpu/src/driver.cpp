// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>
#include<assert.h>

#define BLOCK_SIZE 32 //@@ You can change this
#define REDUX_FACTOR 64
#define ELES_PER_BLOCK 2048
#define HOST_ELES 2048
#define intCeil(a, b) (((a) + (b) - 1) / (b))
#define CAST(expr, type) ((type)(expr))

char *inputFile,*outputFile;
void _errorCheck(cudaError_t e){
	if(e != cudaSuccess){
		printf("Failed to run statement \n");
	}
}

__global__ void total(float *input, float *output, int len) {
	//@@ Compute reduction for a segment of the input vector
	int startIndex = ELES_PER_BLOCK * blockIdx.x + threadIdx.x;
	__shared__ float outputArray[BLOCK_SIZE];

	int localSum = 0;

	#pragma unroll 8
	for (int i = 0; i < REDUX_FACTOR; i++) {
		localSum += input[startIndex + i * BLOCK_SIZE];
		//__syncthreads();
	}
	outputArray[threadIdx.x] = localSum;

	__syncthreads();
	if (threadIdx.x == 0) {
		localSum = 0;
		for (int i = 0; i < BLOCK_SIZE; i++) {
			localSum += outputArray[i];
		}
		output[blockIdx.x] = localSum;
	}
}

void parseInput(int argc, char **argv){
	if(argc < 2){
		printf("Not enough arguments\n");
		printf("Usage: reduction -i inputFile -o outputFile\n");	
		exit(1);
	}
	int i=1;
	while(i<argc){
		if(!strcmp(argv[i],"-i")){
			++i;
			inputFile = argv[i];
		}
		else if(!strcmp(argv[i],"-o")){
			++i;
			outputFile = argv[i];
		}
		else{
			printf("Wrong input");
			exit(1);
		}
		i++;
	}
}
void getSize(int &size, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		perror("Error opening File\n");
		exit(1);
	}
	
	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	fclose(fp);	
}
void readFromFile(int &size,float *v, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		printf("Error opening File %s\n",file);
		exit(1);
	}
	
	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	int i=0;
	float t;
	while(i < size){
		if(fscanf(fp,"%f",&t)==EOF){
			printf("Error reading file\n");
			exit(1);
		}
		v[i++]=t;
	}
	fclose(fp);
	

}

int main(int argc, char **argv) {
	int ii;
	float *hostInput;  // The input 1D list
	float *hostOutput; // The output list
	float *deviceInput;
	float *deviceOutput;
	int numInputElements;  // number of elements in the input list
	int numOutputElements; // number of elements in the output list
	float *solution;
	cudaError_t err;

	// Read arguments and input files
	parseInput(argc,argv);

	// Read input from data
	getSize(numInputElements,inputFile);
	err = cudaMallocHost(CAST(&hostInput, void**), numInputElements * sizeof(float));
	if (err != cudaSuccess) {
		fprintf(stderr, "Could not allocate host memory for input. (%s)\n", cudaGetErrorString(err));
		exit(1);
	}
	hostInput = (float*) malloc(numInputElements*sizeof(float));

	readFromFile(numInputElements,hostInput,inputFile);  

	int opsz;
	getSize(opsz,outputFile);	
	solution = (float*) malloc(opsz*sizeof(float));

	readFromFile(opsz,solution,outputFile);
	int gpuProblems = min(((numInputElements - HOST_ELES) / ELES_PER_BLOCK) * ELES_PER_BLOCK, 0);

	//@@ You can change this, but assumes output element per block
	numOutputElements = gpuProblems / ELES_PER_BLOCK;
	size_t hostOutputSize = min(numOutputElements * sizeof(float), CAST(4, size_t));
	err = cudaMallocHost(CAST(&hostOutput, void**), hostOutputSize);
	if (err != cudaSuccess) {
		fprintf(stderr, "Could not allocate host memory for output. (%s)\n", cudaGetErrorString(err));
		exit(1);
	}


	//@@ Initialize the grid and block dimensions here
	int gridSize = numOutputElements;

	//@@ Allocate GPU memory here
	err = cudaMalloc(CAST(&deviceInput, void**), min(gpuProblems * sizeof(float), CAST(4, size_t)));
	if (err != cudaSuccess) {
	fprintf(stderr, "Could not allocate GPU memory for input (error code %s)!\n", cudaGetErrorString(err));
		printf("Could not allocate GPU memory for input.\n");
		exit(1);
	}
	err = cudaMalloc(CAST(&deviceOutput, void**), min(numOutputElements * sizeof(float), CAST(4, size_t)));
	if (err != cudaSuccess) {
		fprintf(stderr, "Could not allocate GPU memory for output (error code %s)!\n", cudaGetErrorString(err));
		exit(1);
	}

	//@@ Copy memory to the GPU here
	err = cudaMemcpy(deviceInput, hostInput, min(gpuProblems * sizeof(float), CAST(4, size_t)), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy input (error code %s)!\n", cudaGetErrorString(err));
		exit(1);
	}

	// Initialize timer
	cudaEvent_t start,stop;
	float elapsed_time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	if (gridSize > 0) {

		//@@ Launch the GPU Kernel here, you may want multiple implementations to compare
		total<<<gridSize, BLOCK_SIZE>>>(deviceInput, deviceOutput, numInputElements);
		float hostSum = 0;
		// Add the remaining elements using the CPU so we can do less bounds checking.
		// Do this here so we can avoid waiting.
		for (ii = numOutputElements * ELES_PER_BLOCK; ii < numInputElements; ii++) {
			hostSum += hostInput[ii];
		}


		err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch total kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch total kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		//@@ Copy the GPU memory back to the CPU here
		err = cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to copy output (error code %s)!\n", cudaGetErrorString(err));
			exit(1);
		}
		hostOutput[0] += hostSum;

		/*
		* Reduce any remaining output on host
		*/
		for (ii = 1; ii < numOutputElements; ii++) {
			hostOutput[0] += hostOutput[ii];
		}
	} else {
		hostOutput[0] = 0;
		for (ii = numOutputElements * ELES_PER_BLOCK; ii < numInputElements; ii++) {
			hostOutput[0] += hostInput[ii];
		}
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);        
	cudaEventElapsedTime(&elapsed_time,start, stop);

	//@@ Free the GPU memory here
	err = cudaFree(deviceInput);
	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free input memory (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
	}
	err = cudaFree(deviceOutput);
	if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free output memory (error code %s)!\n", cudaGetErrorString(err));
        exit(1);
	}

	if(solution[0] == hostOutput[0]){
		printf("The operation was successful, time = %2.6f\n",elapsed_time);
	}
	else{
		printf("expected: %f actual: %f\n", solution[0], hostOutput[0]);
		printf("The operation failed \n");
	}

	return 0;
}
