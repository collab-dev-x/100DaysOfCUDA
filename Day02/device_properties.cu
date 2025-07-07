#include <iostream>
#include <string>
using namespace std;
#define ERROR_CHECK(call) err_check((call), __LINE__, __FILE__ );

void err_check(cudaError_t call, int line, string file){
    if (call!= cudaSuccess){
        cerr<<"Error at "<<file<<" and "<<line<<" code"<<call<<" \""<<cudaGetErrorString(call)<<"\""<<endl;
        exit(0);
    }
}


int main(){
    int device_count = 0;
    ERROR_CHECK(cudaGetDeviceCount(&device_count));
    if(device_count==0){
        cout<<"No device found\n";
        return 0;
    }
    cout<<"Number of Devices: "<<device_count<<endl;

    for(int i=0; i<device_count;i++){
        cudaDeviceProp prop;
        ERROR_CHECK(cudaGetDeviceProperties(&prop,i));
        cout<<"Nmae of device "<<i<<": "<<prop.name<<endl;
        cout<<"Computer capability: "<<prop.major<<"."<<prop.minor<<endl;
        cout<<"Clock Rate: "<<prop.clockRate<<endl;
        cout<<"Device copy overlap: "<<(prop.deviceOverlap?"Enabled":"Disabled")<<endl;
        cout<<"Number of copy engines: "<<prop.asyncEngineCount<<endl;
        cout<<"Concurrent Kernels: "<<(prop.concurrentKernels?"Available":"Unavailable")<<endl;
        cout<<"Kernel execution timeout: "<<(prop.kernelExecTimeoutEnabled?"Enabled":"Disabled")<<endl;
        cout<<"Global Memory: "<<prop.totalGlobalMem<<endl;
        cout<<"Total Constant Memory: "<<prop.totalConstMem<<endl;
        cout<<"Max memory pitch: "<<prop.memPitch<<endl;
        cout<<"Texture alignment: "<<prop.textureAlignment<<endl;
        cout<<"Multiprocessor count: "<<prop.multiProcessorCount<<endl;
        cout<<"Max blocks per processor: "<<prop.maxBlocksPerMultiProcessor<<endl;
        cout<<"Shared Memory per processor: "<<prop.sharedMemPerMultiprocessor<<endl;
        cout<<"Registers per processor: "<<prop.regsPerBlock<<endl;
        cout<<"Threads in warp: "<<prop.warpSize<<endl;
        cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<endl;
        cout<<"Max thread dimensions: "<<prop.maxThreadsDim[0]<<","<<prop.maxThreadsDim[1]<<","<<prop.maxThreadsDim[2]<<endl;
        cout<<"Max grid dimensions: "<<prop.maxGridSize[0]<<","<<prop.maxGridSize[1]<<","<<prop.maxGridSize[2]<<endl<<endl;
        cout<<"Error Correcting Code: "<<(prop.ECCEnabled?"Enabled":"Disabled")<<endl;
    
    }

}