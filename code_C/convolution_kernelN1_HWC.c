const int16_t height[6] =  {32,16,8,144,176,96};
const int16_t IC[6] =  {3,32,32,3,3,3};
const int16_t OC[6] =  {32,32,64,16,16,16};
const int16_t OH[6] =  {32,16,8,72,88,48};
const int16_t PADDING[6] =  {2,2,2,1,1,1};
const int16_t SH[6] =  {1,1,1,2,2,2};
const int16_t KH[6] =  {5,5,5,3,3,3};
const int32_t output[6] =  {3072,8192,2048,62208,92928,27648};
const int32_t bufferA[6] =  {35840,16384,6144,145152,216832,64512};
const signed char CHWweight[51200];
const int32_t bias[64];
static signed char buffer[216940];
