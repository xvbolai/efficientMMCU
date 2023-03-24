const int16_t height[6] =  {32,16,8,5,6,3};
const int16_t IC[6] =  {32,32,64,448,384,160};
const int16_t OH[6] =  {16,8,4,1,1,1};
const int16_t PADDING[6] =  {0,0,0,0,0,0};
const int16_t SH[6] =  {2,2,2,5,6,3};
const int16_t KH[6] =  {2,2,2,5,6,3};
const int32_t output[6] =  {32768,8192,4096,11200,13824,1440};
const int32_t bufferA[6] =  {40960,10240,5120,11648,14208,1600};
static signed char buffer[41984];
