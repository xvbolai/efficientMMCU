const int16_t height[63] =  {72,72,36,36,18,18,18,9,9,9,9,9,9,5,5,5,5,5,88,88,44,44,44,22,22,22,11,11,11,11,11,11,11,11,6,6,6,6,6,6,48,48,24,24,24,12,12,12,6,6,6,6,6,6,6,6,6,3,3,3,3,3,3};
const int16_t IC[63] =  {16,8,48,8,48,16,96,96,24,144,144,32,192,192,56,336,336,112,16,8,24,16,48,48,16,16,96,24,72,24,144,32,96,32,192,64,384,64,384,96,16,8,32,16,48,48,24,120,120,40,160,160,48,144,48,192,48,240,96,384,96,576,160};
const int16_t OC[63] =  {8,48,8,48,16,96,16,24,144,24,32,192,32,56,336,56,112,448,8,24,16,48,16,16,48,96,24,72,24,144,32,96,32,192,64,384,64,192,96,384,8,32,16,48,16,24,120,24,40,160,40,48,144,48,192,48,240,96,384,96,576,160,160};
const int16_t OH[63] =  {72,72,36,36,18,18,18,9,9,9,9,9,9,5,5,5,5,5,88,88,44,44,44,22,22,22,11,11,11,11,11,11,11,11,6,6,6,6,6,6,48,48,24,24,24,12,12,12,6,6,6,6,6,6,6,6,6,3,3,3,3,3,3};
const int16_t PADDING[63] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
const int16_t SH[63] =  {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const int16_t KH[63] =  {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
const int32_t output[63] =  {82944,41472,62208,10368,15552,5184,31104,7776,1944,11664,11664,2592,15552,4800,1400,8400,8400,2800,123904,61952,46464,30976,92928,23232,7744,7744,11616,2904,8712,2904,17424,3872,11616,3872,6912,2304,13824,2304,13824,3456,36864,18432,18432,9216,27648,6912,3456,17280,4320,1440,5760,5760,1728,5184,1728,6912,1728,2160,864,3456,864,5184,1440};
const int32_t bufferA[63] =  {124416,290304,72576,72576,20736,36288,36288,9720,13608,13608,14256,18144,18144,6200,9800,9800,11200,14000,185856,247808,77440,123904,123904,30976,30976,54208,14520,11616,11616,20328,21296,15488,15488,27104,9216,16128,16128,9216,17280,17280,55296,92160,27648,36864,36864,10368,20736,20736,5760,7200,7200,7488,6912,6912,8640,8640,10368,3024,4320,4320,6048,6624,2880};
const signed char CHWweight[92160];
const int32_t bias[576];
static signed char buffer[290336];