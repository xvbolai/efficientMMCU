import pandas as pd 
import numpy as np

# df = pd.read_csv('dataset_conv2d_large_cnn.csv')
# df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
# df = df[df['GP'] != df['IC']]
# df = df[df['KH'] != 1]
# df.to_csv('convolution.csv',index = False ,sep = ',')

def add_str(file, str):
     with open('./code_C/' + file + '.c', "a") as f:
        f.write(str + '\n')

def save_param(data, file, name, len, type):
    with open('./code_C/' + file + '.c', "a") as f:
        f.write(f"{type} {name}[{len}] = " + " {")
        data.tofile(f, sep=",")
        f.write("};\n")

def depth_wise_convolution_3x3_stride1_CHW(data):
    data = data[data['GP'] == data['IC']]
    data = data[data['KH'] == 3]
    data = data[data['SH'] == 1]
    # input and ouput memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
   
    # data['ouput'] = ((data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1) * ((data['IW'] + 2 * data['PW'] - data['KW']) // data['SW'] + 1) * data['OC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    
    # weight size 
    data['CHWweight'] = data['KH'] * data['KW'] * data['IC']
    # bias size
    data['offsetBias'] = data['OC']
    data['offsetRBias'] = data['OC']
    # multi size
    data['scale'] = data['OC']
    # immediate size
    data['sbuf'] = (data['IH'] + 2 * data['PH']) * (data['IW'] + 2 * data['PW'])
    # buff size
    # data['buffer'] = data['input'] + data['sbuf']
    
    
    length = len(data)
    max_buff = data['input'].max() + data['sbuf'].max() # max buff size
    max_input = data['input'].max() # max input size
    file = 'depthwise_kernel3x3_stride1_CHW'
    # weight
    # save_param(data['CHWweight'].to_numpy(), file, 'CHWweight', length, 'static signed char') # weight size array
    CHWweight = data['CHWweight'].max()
    # bias
    # save_param(data['offsetBias'].to_numpy(), file, 'offsetBias', length, 'const int32_t') 
    offsetBias = data['offsetBias'].max()
    
    # save_param(data['offsetRBias'].to_numpy(), file, 'offsetRBias', length, 'const int32_t') 
    offsetRBias = data['offsetRBias'].max()
    # scale
    scale = data['scale'].max()
    # save_param(data['scale'].to_numpy(), file, 'scale', length, 'const float')
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel 
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t')
    # output channel
    save_param(data['OC'].to_numpy(),  file, 'OC', length, 'const int16_t')
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    # save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int32_t')
    # CHWweight
    add_str(file, f'const signed char CHWweight[{CHWweight}];')
    # offsetBias
    add_str(file, f'const int32_t offsetBias[{offsetBias}];')
    # offsetRBias
    add_str(file, f'const int32_t offsetRBias[{offsetRBias}];')
    # scale
    add_str(file, f'const float scale[{scale}];')
    
    add_str(file, f'static signed char buffer[{max_buff}];')
    add_str(file, f'static int16_t *sbuf = (int16_t *)&buffer[{max_input}];')

def depth_wise_convolution_3x3_stride2_CHW(data):
    data = data[data['GP'] == data['IC']]
    data = data[data['KH'] == 3]
    data = data[data['SH'] == 2]
    # input and ouput memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
   
    # data['ouput'] = ((data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1) * ((data['IW'] + 2 * data['PW'] - data['KW']) // data['SW'] + 1) * data['OC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    
    # weight size 
    data['CHWweight'] = data['KH'] * data['KW'] * data['IC']
    # bias size
    data['offsetBias'] = data['OC']
    data['offsetRBias'] = data['OC']
    # multi size
    data['scale'] = data['OC']
    # immediate size
    data['sbuf'] = (data['IH'] + 2 * data['PH']) * (data['IW'] + 2 * data['PW'])
    # buff size
    # data['buffer'] = data['input'] + data['sbuf']
    
    
    length = len(data)
    max_buff = data['input'].max() + data['sbuf'].max() # max buff size
    max_input = data['input'].max() # max input size
    file = 'depthwise_kernel3x3_stride2_CHW'
    # weight
    # save_param(data['CHWweight'].to_numpy(), file, 'CHWweight', length, 'static signed char') # weight size array
    CHWweight = data['CHWweight'].max()
    # bias
    # save_param(data['offsetBias'].to_numpy(), file, 'offsetBias', length, 'const int32_t') 
    offsetBias = data['offsetBias'].max()
    
    # save_param(data['offsetRBias'].to_numpy(), file, 'offsetRBias', length, 'const int32_t') 
    offsetRBias = data['offsetRBias'].max()
    # scale
    scale = data['scale'].max()
    # save_param(data['scale'].to_numpy(), file, 'scale', length, 'const float')
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel 
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t')
    # output channel
    save_param(data['OC'].to_numpy(),  file, 'OC', length, 'const int16_t')
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    # save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int32_t')
    # CHWweight
    add_str(file, f'const signed char CHWweight[{CHWweight}];')
    # offsetBias
    add_str(file, f'const int32_t offsetBias[{offsetBias}];')
    # offsetRBias
    add_str(file, f'const int32_t offsetRBias[{offsetRBias}];')
    # scale
    add_str(file, f'const float scale[{scale}];')
    
    add_str(file, f'static signed char buffer[{max_buff}];')
    add_str(file, f'static int16_t *sbuf = (int16_t *)&buffer[{max_input}];')
    
def depth_wise_convolution_HWC(data):
    data = data[data['GP'] == data['IC']].copy()
    # input and ouput memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
   
    # data['ouput'] = ((data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1) * ((data['IW'] + 2 * data['PW'] - data['KW']) // data['SW'] + 1) * data['OC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    # ouput
    data['output'] = data['OH'] * data['OH'] * data['OC']
    # weight size 
    data['CHWweight'] = data['KH'] * data['KW'] * data['IC']
    # bias size
    data['bias'] = data['OC']
    data['bufferA'] = 4*data['IC']*data['KH']*data['KW']
    
    # max_buff = data['input'].max() + data['output'].max() + data['bufferA'].max()# max buff size
    data['buffer'] = data['input'] + data['output'] + data['bufferA']
    data = data[data['buffer'] < 293319].copy()
    length = len(data)
    max_buff = data['buffer'].max()
    file = 'depthwise_kernel_HWC'
    # weight
    # save_param(data['CHWweight'].to_numpy(), file, 'CHWweight', length, 'static signed char') # weight size array
    CHWweight = data['CHWweight'].max()
    # bias
    bias = data['bias'].max()
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel 
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t')
    # output channel
    save_param(data['OC'].to_numpy(),  file, 'OC', length, 'const int16_t')
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int16_t')
    # stride
    save_param(data['SH'].to_numpy(),  file, 'SH', length, 'const int16_t')
    # kernel size
    save_param(data['KH'].to_numpy(),  file, 'KH', length, 'const int16_t')
    save_param(data['input'].to_numpy(),  file, 'output', length, 'const int32_t')
    # bufferA
    save_param((data['input'] + data['output']).to_numpy(),  file, 'bufferA', length, 'const int32_t')
    
    # CHWweight
    add_str(file, f'const signed char CHWweight[{CHWweight}] = {{0}};')
    # bias
    add_str(file, f'const int32_t bias[{bias}] = {{0}};')
    add_str(file, f'static signed char buffer[{max_buff}];')

def conv_kernel1_stride1_pad0_HWC(data):
    data = data[data['GP'] != data['IC']]
    data = data[data['OC'] != 1000]
    data = data[data['KH'] == 1].copy()

    # input and ouput memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
   
    # data['ouput'] = ((data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1) * ((data['IW'] + 2 * data['PW'] - data['KW']) // data['SW'] + 1) * data['OC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    # ouput
    data['output'] = data['OH'] * data['OH'] * data['OC']
    # weight size 
    data['CHWweight'] = data['KH'] * data['KW'] * data['IC']*data['OC']
    # bias size
    data['bias'] = data['OC']
    data['bufferA'] = 4*data['IC']*data['KH']*data['KW']
    
    
    length = len(data)
    # max_input = data['input'].max() # max input size
    # max_output = data['output'].max()
    data['buffer'] = data['input'] + data['output'] + data['bufferA']
    max_buff = data['buffer'].max()
   
    file = 'convolution_kernel1_stride1_pad0_HWC'
    # weight
    # save_param(data['CHWweight'].to_numpy(), file, 'CHWweight', length, 'static signed char') # weight size array
    CHWweight = data['CHWweight'].max()
    # bias
    bias = data['bias'].max()
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t') 
    # output channel
    save_param(data['OC'].to_numpy(),  file, 'OC', length, 'const int16_t')
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int16_t')
    # stride
    save_param(data['SH'].to_numpy(),  file, 'SH', length, 'const int16_t')
    # kernel size
    save_param(data['KH'].to_numpy(),  file, 'KH', length, 'const int16_t')
    # input size array
    save_param(data['input'].to_numpy(),  file, 'output', length, 'const int32_t')
    # bufferA
    save_param((data['input'] + data['output']).to_numpy(),  file, 'bufferA', length, 'const int32_t')
    # CHWweight
    add_str(file, f'const signed char CHWweight[{CHWweight}];')
    # bias
    add_str(file, f'const int32_t bias[{bias}];')
    add_str(file, f'static signed char buffer[{max_buff}];')
    
def conv_kernelN1_HWC(data):
    data = data[data['GP'] != data['IC']]
    data = data[data['KH'] != 1].copy()

    # input and output memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
   
    # data['ouput'] = ((data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1) * ((data['IW'] + 2 * data['PW'] - data['KW']) // data['SW'] + 1) * data['OC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    # output
    data['output'] = data['OH'] * data['OH'] * data['OC']
    # weight size 
    data['CHWweight'] = data['KH'] * data['KH'] * data['IC']*data['OC']
    # bias size
    data['bias'] = data['OC']
    data['bufferA'] = 4*data['IC']*data['KH']*data['KW']
    
    
    length = len(data)
    # max_input = data['input'].max() # max input size
    # max_output = data['output'].max()
    data['buffer'] = data['input'] + data['output'] + data['bufferA']
    max_buff = data['buffer'].max()
   
    file = 'convolution_kernelN1_HWC'
    # weight
    # save_param(data['CHWweight'].to_numpy(), file, 'CHWweight', length, 'static signed char') # weight size array
    CHWweight = data['CHWweight'].max()
    # bias
    bias = data['bias'].max()
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t') 
    # output channel
    save_param(data['OC'].to_numpy(),  file, 'OC', length, 'const int16_t')
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int16_t')
    # stride
    save_param(data['SH'].to_numpy(),  file, 'SH', length, 'const int16_t')
    # kernel size
    save_param(data['KH'].to_numpy(),  file, 'KH', length, 'const int16_t')
    # input size array
    save_param(data['input'].to_numpy(),  file, 'output', length, 'const int32_t')
    # bufferA
    save_param((data['input'] + data['output']).to_numpy(),  file, 'bufferA', length, 'const int32_t')
    # CHWweight
    add_str(file, f'const signed char CHWweight[{CHWweight}];')
    # bias
    add_str(file, f'const int32_t bias[{bias}];')
    add_str(file, f'static signed char buffer[{max_buff}];')

def conversion_CHW_to_HWC(data):
    pass

def conversion_HWC_to_CHW(data):
    pass

def conversion_sync(data):
    pass

def max_pool_HWC(data):
    pass

def avg_pool_HWC(data):
    # input and output memory size
    data['input'] = data['IH'] * data['IW'] * data['IC']
    # output height
    data['OH'] = (data['IH'] + 2 * data['PH'] - data['KH']) // data['SH'] + 1
    # output
    data['output'] = data['OH'] * data['OH'] * data['OC']
    # 2*dim_im_out*ch_im_in
    data['bufferA'] = 2*data['OH']*data['IC']
    
    length = len(data)
    # max_input = data['input'].max() # max input size
    # max_output = data['output'].max()
    data['buffer'] = data['input'] + data['output'] + data['bufferA']
    max_buff = data['buffer'].max()
   
    file = 'arm_avepool_q7_HWC'
    # height
    save_param(data['IH'].to_numpy(),  file, 'height', length, 'const int16_t')
    # input channel
    save_param(data['IC'].to_numpy(),  file, 'IC', length, 'const int16_t') 
    # ouput height
    save_param(data['OH'].to_numpy(),  file, 'OH', length, 'const int16_t')
    # padding
    save_param(data['PH'].to_numpy(),  file, 'PADDING', length, 'const int16_t')
    # stride
    save_param(data['SH'].to_numpy(),  file, 'SH', length, 'const int16_t')
    # kernel size
    save_param(data['KH'].to_numpy(),  file, 'KH', length, 'const int16_t')
    # input size array
    save_param(data['input'].to_numpy(),  file, 'output', length, 'const int32_t')
    # bufferA
    save_param((data['input'] + data['output']).to_numpy(),  file, 'bufferA', length, 'const int32_t')
    
    add_str(file, f'static signed char buffer[{max_buff}];')


if __name__ == '__main__':
    df = pd.read_csv('./op_datasets/dataset_conv2d_large_cnn.csv')
    df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
    # CHW depth-wise conv kernel size 3x3, stride 1x1, padding 1x1
    depth_wise_convolution_3x3_stride1_CHW(df)
    # CHW depth-wise conv kernel size 3x3, stride 2x2, padding 1x1
    depth_wise_convolution_3x3_stride2_CHW(df)
    # HWC depth-wsie 
    depth_wise_convolution_HWC(df)
    # HWC convolution kernel 1x1 stride 1 padding 0
    conv_kernel1_stride1_pad0_HWC(df)
    # HWC convolution kernel N1
    conv_kernelN1_HWC(df)
    
    # dataset_pooling_large_cnn.csv
    df = pd.read_csv('./op_datasets/dataset_pooling_large_cnn.csv')
    df = df.drop_duplicates(subset = None, keep ='first', inplace = False)
    avg_pool_HWC(df)