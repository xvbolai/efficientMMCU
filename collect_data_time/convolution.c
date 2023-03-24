//q15_t IH[75] =  {32,14,112,112,28,28,28,28,28,14,28,14,14,14,14,7,112,56,56,28,28,28,28,14,14,14,14,14,14,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,4,4,4,4,4,4,112,26,26,26,26,26,26,26,26,26,13,13,13,13,13,13,13,13,13,13,13,13,6,6,6};
//q15_t IC[75] =  {3,16,3,3,64,64,64,64,64,128,64,128,128,128,128,256,3,32,16,96,24,144,24,144,32,192,32,192,32,192,64,384,64,384,64,384,64,384,96,576,96,576,96,576,160,960,160,960,160,960,3,96,16,16,128,16,16,128,32,32,256,32,32,256,48,48,384,48,48,384,64,64,512,64,64};
//q15_t OC[75] =  {16,32,96,64,64,64,64,64,128,128,128,128,128,256,256,512,32,16,96,24,144,24,144,32,192,32,192,32,192,64,384,64,384,64,384,64,384,96,576,96,576,96,576,160,960,160,960,160,960,320,96,16,64,64,16,64,64,32,128,128,32,128,128,48,192,192,48,192,192,64,256,256,64,256,256};
//q15_t KH[75] =  {5,5,11,7,3,3,3,3,3,3,1,3,3,3,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3,1,1,3};
//q15_t SH[75] =  {1,1,4,2,1,1,1,1,2,1,2,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//q15_t PH[75] =  {0,0,2,3,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1};
//q15_t OH[75] =  {28,10,27,56,28,28,28,28,14,14,14,14,14,7,7,4,56,56,56,28,28,28,28,14,14,14,14,14,14,7,7,7,7,7,7,7,7,7,7,7,7,7,7,4,4,4,4,4,4,4,53,26,26,26,26,26,26,26,26,26,13,13,13,13,13,13,13,13,13,13,13,13,6,6,6};
//q7_t in[112896];
//q7_t out[5376];
//q15_t bufferA[2304];
//q7_t weight[307200];
//q7_t bias[960];

//	  for(int k = 0; k < 15; ++k) {
//		  depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-128,128,-128,127,&buffer0[0],72,72,16,sbuf,-128);
//		  depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,-128,128,-128,127,&buffer0[0],9,9,192,sbuf,-128);
//	  }
	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],36,36,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-128,128,-128,127,&buffer0[0],36,36,48,sbuf,-128);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,scales13,-128,128,-128,127,&buffer0[0],18,18,96,sbuf,-128);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,-128,128,-128,127,&buffer0[0],9,9,192,sbuf,-128);
//	  depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],72,72,48,(const q7_t*) CHWweight4,offsetBias4,offsetRBias4,scales4,-128,128,-128,127,&buffer0[0],36,36,48,sbuf,-128);
//	  depthwise_kernel3x3_stride2_inplace_CHW_fpreq(&buffer0[0],9,9,192,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,scales40,-128,128,-128,127,&buffer0[0],5,5,192,sbuf,-128);

//	  end = HAL_GetTick() - start; // ms

//	  printf("depthwise_kernel3x3_stride1_inplace_CHW_fpreq HWC time %d\r\n", end);
//	  start = HAL_GetTick();
//	  arm_depthwise_separable_conv_HWC_q7(&buffer0[0], 72, 16, (const q7_t*) CHWweight1, 16, 3, 1, 1, offsetBias1, 8, 8, &out[0], 72, sbuf, (const q7_t*)NULL);
//	  for(int k = 0; k < 15; ++k) {
//		  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-128,128,-128,127,&buffer0[0],72,72,16,sbuf,-128);
//		  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],9,9,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,-128,128,-128,127,&buffer0[0],9,9,192,sbuf,-128);
//	  }

//	  end = HAL_GetTick() - start;
//	  printf("depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy CHW time %d\r\n", end);
//	  start = HAL_GetTick();

//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],44,44,80,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-128,128,-128,127,&buffer0[0],44,44,80,sbuf,-128);
//	  printf("start\r\n");
//	  for(int i = 44; i >= 6; --i){
//		  for(int c = 80; c >= 16; --c){
//			  start = HAL_GetTick();
//	//		  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],i,i,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-128,128,-128,127,&buffer0[0],i,i,16,sbuf,-128);
//			  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],i,i,c,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-128,128,-128,127,&buffer0[0],i,i,c,sbuf,-128);
//			  end1 = HAL_GetTick() - start;
//			  start = HAL_GetTick();
//			  arm_depthwise_separable_conv_HWC_q7(&buffer0[0], i, c, (const q7_t*) CHWweight7, c, 3, 1, 1, offsetBias7, 8, 8, &out[0], i, sbuf, (const q7_t*)NULL);
//			  end2 = HAL_GetTick() - start;
//			  printf("%d,%d,%d,%d,%d\r\n", i, c, 19 * c * i * i, end1, end2);
//		  }
//	  }

//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],88,88,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-128,128,-128,127,&buffer0[0],88,88,16,sbuf,-128);
//	  arm_depthwise_separable_conv_HWC_q7(&buffer0[0], 9, 192, (const q7_t*) CHWweight34, 192, 3, 1, 1, offsetBias34, 8, 8, &out[0], 9, sbuf, (const q7_t*)NULL);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],72,72,16,(const q7_t*) CHWweight1,offsetBias1,offsetRBias1,scales1,-128,128,-128,127,&buffer0[0],72,72,16,sbuf,-128);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],9,9,192,(const q7_t*) CHWweight34,offsetBias34,offsetRBias34,scales34,-128,128,-128,127,&buffer0[0],9,9,192,sbuf,-128);

//	  end = HAL_GetTick() - start;
//	  printf("9x9x192 CHW time %d\r\n", end);
//	  start = HAL_GetTick();
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],11,11,240,(const q7_t*) CHWweight40,offsetBias40,offsetRBias40,scales40,-128,128,-128,127,&buffer0[0],11,11,240,sbuf,-128);

//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],44,44,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-128,128,-128,127,&buffer0[0],44,44,48,sbuf,-128);
//	  arm_depthwise_separable_conv_HWC_q7(&buffer0[0], 9, 144, (const q7_t*) CHWweight25, 144, 3, 1, 1, offsetBias25, 8, 8, &out[0], 9, sbuf, (const q7_t*)NULL);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],36,36,48,(const q7_t*) CHWweight7,offsetBias7,offsetRBias7,scales7,-128,128,-128,127,&buffer0[0],36,36,48,sbuf,-128);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],9,9,144,(const q7_t*) CHWweight25,offsetBias25,offsetRBias25,scales25,-128,128,-128,127,&buffer0[0],9,9,144,sbuf,-128);

//	  end = HAL_GetTick() - start;
//	  printf("9x9x144 CHW time %d\r\n", end);
//	  start = HAL_GetTick();
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],6,6,384,(const q7_t*) CHWweight49,offsetBias49,offsetRBias49,scales49,-128,128,-128,127,&buffer0[0],6,6,384,sbuf,-128);

//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],22,22,48,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,scales13,-128,128,-128,127,&buffer0[0],22,22,48,sbuf,-128);
//	  arm_depthwise_separable_conv_HWC_q7(&buffer0[0], 5, 336, (const q7_t*) CHWweight43, 336, 3, 1, 1, offsetBias43, 8, 8, &out[0], 5, sbuf, (const q7_t*)NULL);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],18,18,96,(const q7_t*) CHWweight13,offsetBias13,offsetRBias13,scales13,-128,128,-128,127,&buffer0[0],18,18,96,sbuf,-128);
//	  depthwise_kernel3x3_stride1_inplace_CHW_fpreq_memcpy(&buffer0[0],5,5,336,(const q7_t*) CHWweight43,offsetBias43,offsetRBias43,scales43,-128,128,-128,127,&buffer0[0],5,5,336,sbuf,-128);

//	  end = HAL_GetTick() - start;
//	  printf("5x5x336 CHW time %d\r\n", end);
	  for(int i = 0; i < 75; ++i) {
		  start = HAL_GetTick();
		  arm_convolve_HWC_q7_basic_nonsquare(in, IH[i],IH[i],IC[i], weight, OC[i], KH[i], KH[i], PH[i], PH[i], SH[i], SH[i], bias, 8, 8, out,  OH[i], OH[i], bufferA, (q7_t *)NULL);
		  end1 = HAL_GetTick() - start;

		  start = HAL_GetTick();
//		  CHW2HWC(in, weight, IH[i],IH[i],IC[i]);
		  arm_convolve_HWC_q7_basic_nonsquare_CHW(in, IH[i],IH[i],IC[i], weight, OC[i], KH[i], KH[i], PH[i], PH[i], SH[i], SH[i], bias, 8, 8, out,  OH[i], OH[i], bufferA, (q7_t *)NULL);
		  HWC2CHW(out, in, IH[i],IH[i],IC[i]);
		  end2 = HAL_GetTick() - start;
		  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\r\n", IH[i],IH[i],IC[i],OC[i],PH[i],PH[i],KH[i],KH[i],SH[i],SH[i], OH[i], OH[i],end1, end2);
	  }