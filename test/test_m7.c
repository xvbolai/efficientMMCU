//const int32_t height[6] =  {32,16,8,144,176,96};
//const int32_t IC[6] =  {3,32,32,3,3,3};
//const int32_t OC[6] =  {32,32,64,16,16,16};
//const int32_t OH[6] =  {32,16,8,72,88,48};
//const int32_t PADDING[6] =  {2,2,2,1,1,1};
//const int32_t SH[6] =  {1,1,1,2,2,2};
//const int32_t KH[6] =  {5,5,5,3,3,3};
//const int32_t output[6] =  {3072,8192,2048,62208,92928,27648};
//const int32_t bufferA[6] =  {35840,16384,6144,145152,216832,64512};
//const signed char CHWweight[51200];
//const int32_t bias[64];
//static signed char buffer[216940];

static int16_t outputchannel;


const int32_t height[6] =  {32,16,8,5,6,3};
const int32_t IC[6] =  {32,32,64,448,384,160};
const int32_t OH[6] =  {16,8,4,1,1,1};
const int32_t PADDING[6] =  {0,0,0,0,0,0};
const int32_t SH[6] =  {2,2,2,5,6,3};
const int32_t KH[6] =  {2,2,2,5,6,3};
const int32_t output[6] =  {32768,8192,4096,11200,13824,1440};
const int32_t bufferA[6] =  {40960,10240,5120,11648,14208,1600};
static signed char buffer[41984];

printf("start\r\n");
for(int i = 0; i < 6; ++i){
//		  for(int j = 1; j <= 10; ++j) {
//			  //CHW-style format
//			  outputchannel = OH[i] * j * 0.1;
//			  start = HAL_GetTick();
//			  for(int k = 0; k  < 15; ++k)
////				  arm_convolve_1x1_HWC_q7_fast_nonsquare(&buffer[0], height[i], height[i], IC[i], CHWweight, OC[i], KH[i], KH[i], PADDING[i], PADDING[i], SH[i], SH[i], bias, 8, 8, buffer + output[i], OH[i], outputchannel, buffer + bufferA[i], (q7_t *)NULL);
//				  arm_convolve_HWC_q7_basic_nonsquare(&buffer[0], height[i], height[i], IC[i], CHWweight, OC[i], KH[i], KH[i], PADDING[i], PADDING[i], SH[i], SH[i], bias, 8, 8, buffer + output[i], OH[i], outputchannel, buffer + bufferA[i], (q7_t *)NULL);
////				  arm_depthwise_separable_conv_HWC_q7_nonsquare(&buffer[0], height[i], height[i], OC[i], CHWweight, OC[i], KH[i], KH[i], PADDING[i], PADDING[i], SH[i], SH[i], bias, 8, 8, output, OH[i], outputchannel, bufferA, (q7_t *)NULL);
////				  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer[0],height[i],height[i],outputchannel,(const q7_t*) CHWweight,offsetBias,offsetRBias,scale,-128,128,-128,127,&buffer[0],OH[i],OH[i],outputchannel,sbuf,-128);
////				  depthwise_kernel3x3_stride2_inplace_CHW_fpreq_memcpy(&buffer[0],height[i],height[i],outputchannel,(const q7_t*) CHWweight,offsetBias,offsetRBias,scale,-128,128,-128,127,&buffer[0],OH[i],OH[i],outputchannel,sbuf,-128);
//			  end = HAL_GetTick() - start; // ms
//			  //IH,IW,IC,OC,PH,PW,KH,KW,SH,SW,OH,OW,T
//			  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\r\n", height[i],height[i],IC[i],OC[i],PADDING[i],PADDING[i],KH[i],KH[i],SH[i],SH[i], outputchannel, OH[i], end);
//		  }
    for(int j = 1; j <= 10; ++j) {
        outputchannel = IC[i] * j * 0.1;
        start = HAL_GetTick();
        for(int k = 0; k < 25; ++k)
            arm_avepool_q7_HWC(&buffer[0], height[i],outputchannel, KH[i], PADDING[i], SH[i], OH[i], &buffer[0] + bufferA[i], &buffer[0] + output[i]);
        end = HAL_GetTick() - start; // ms
        //IH,IW,IC,OC,PH,PW,KH,KW,SH,SW,OH,OW,T
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\r\n", height[i],height[i],outputchannel,outputchannel,PADDING[i],PADDING[i],KH[i],KH[i],SH[i],SH[i], OH[i], OH[i], end);
    }
}


/* start the execution */
for(int idex =0; idex < 54; ++idex)
{
    for(int i = 1; i <= 10; ++i){
        start = HAL_GetTick();
        arm_convolve_HWC_q7_general_nonsquare(in, IH[idex], IH[idex], IC[idex], weight, OC[idex], KH[idex], KH[idex], PH[idex], PH[idex], SH[idex], SH[idex], bias, 8, 8, out, OH[idex] * i * 0.1, OH[idex], bufferA, (q7_t *)NULL);
        end = HAL_GetTick() - start;
        p[0] = IH[idex],
        p[1] = IC[idex];
        p[2] = OC[idex];
        p[3] = PH[idex];
        p[4] = KH[idex];
        p[5] = SH[idex];
        p[6] = (int)(OH[idex] * i * 0.1);
        p[7] = OH[idex];
        p[8] = end;
        p[9] = i;
//			  arm_fully_connected_q7_opt(in, weight, dim_vec[idex], num_of_rows[idex] * i * 0.1, 8, 8, bias, out, bufferA);
//
//			  end = HAL_GetTick() - start;
//			  p[0] = (int)(num_of_rows[idex] * i * 0.1);
//			  p[1] = dim_vec[idex];
//			  p[2] = end;
    }
}

//	  if(pre != p[13]) {
//		  pre = p[13];
//		  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\r\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12]);
//	  }