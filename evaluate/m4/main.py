from metric import PM10ACC
import pandas as pd



def convolution(name = 'MnasNet'):
    
    df = pd.read_csv(f'./conv/convolution_kernel1_HWC_15time_{name}.csv')
    # df = df[df['OH'] == df['OW']]
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    
    # lable 
    # x_names = ['M', 'K', 'N', 'MK', 'KN', 'MN', 'MKN']
    # target = 'T'
    # 0.024230564904017492, -0.00453348548273885, -0.007127751375797178, 0.000808780126067989, 0.00040477039273403945, 0.006018592037438807, 0.0006263122380181867], "inter": 1.8305150976430582
    
    df['predict'] = 0.024230564904017492*df['M'] -0.00453348548273885*df['K'] -0.007127751375797178*df['N'] +0.000808780126067989*df['MK'] +0.00040477039273403945*df['KN'] +0.006018592037438807*df['MN'] +0.0006263122380181867*df['MKN'] +1.8305150976430582
    acc = PM10ACC(df['predict'].to_numpy(), df['T'].to_numpy())
    print(acc)
    
    number = len(df['predict'])
    lr_true_count = int(number * acc)
    # number = 0
    # lr_true_count = 0
    df = pd.read_csv(f'./conv/convolution_kernelN1_HWC_15time_{name}.csv')
    # df = df[df['OH'] == df['OW']]
    df['FLOPs'] = df['OH'] * df['OW'] * df['OC'] * (2 * df['KH'] * df['KW'] * df['IC'] + 1)
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    # df['FLOPs_t'] = df['T'] / df['FLOPs']
    df['M'] = df['OH'] * df['OW']
    df['K'] = df['IC'] * df['KH'] * df['KW']
    df['N'] = df['OC']
    df['MN'] = df['M'] * df['N']
    df['KN'] = df['K'] * df['N']
    df['MK'] = df['M'] * df['K']
    df['MKN'] = df['M'] * df['K'] * df['N']
    df['predict'] = -0.17381468205212586*df['M'] + 0.022317801768791234*df['K'] -0.004158266088162003*df['N'] +0.0018511850328215196*df['MK'] -0.00024004145174525988*df['KN'] + 0.028897977587329045*df['MN'] +0.0007404582257558425*df['MKN'] -2.215798275217594
    # -0.17381468205212586, 0.022317801768791234, -0.004158266088162003, 0.0018511850328215196, -0.00024004145174525988, 0.028897977587329045, 0.0007404582257558425], "inter": -2.215798275217594}
    acc = PM10ACC(df['predict'].to_numpy(), df['T'].to_numpy())
    print(acc)
    lr_true_count = int(len(df['predict']) * acc) + lr_true_count
    number = number + len(df['predict'])
    print(lr_true_count / number * 100)
    print(lr_true_count, number)

def depthwise_convolution(name = 'MnasNet'):
    
    # 7.907782380376356e-08, 5.0050820738844754e-06, -1.071887044675417e-07], "inter": 0.001857779969703545
    
    df = pd.read_csv(f'./depthwise_conv/depthwise_conv_{name}.csv')
    # FLOPs
    # df = df[df['OH'] == df['OW']]
    df['FLOPs'] = (2 * df['KH'] * df['KW'] + 1) * df['OH'] * df['OW'] * df['OC']
    # filter
    df = df[df['FLOPs'] > 3]
    df = df[df['T'] > 40]
    # per OP time
    df['FLOPs_t'] = df['T'] / df['FLOPs']
     # lable 
    # x_names = ['OH','OW', 'OC']
    # target = 'FLOPs_t'
    df['predict'] = 7.907782380376356e-08*df['OH'] +5.0050820738844754e-06*df['OW'] -1.071887044675417e-07*df['OC'] + 0.001857779969703545
    # -0.17381468205212586, 0.022317801768791234, -0.004158266088162003, 0.0018511850328215196, -0.00024004145174525988, 0.028897977587329045, 0.0007404582257558425], "inter": -2.215798275217594}
    acc = PM10ACC(df['predict'].to_numpy(), df['FLOPs_t'].to_numpy())
    
    number =  len(df['predict'])
    lr_true_count = int(len(df['predict']) * acc)
    
    print(lr_true_count, number)
    
depthwise_convolution(name = 'Proxyless')
