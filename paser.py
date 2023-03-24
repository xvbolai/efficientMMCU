import  collections
import torch.nn as nn
import torch
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp
import json
from functools import *
from operator import mul

DEVICE = 0.6
RATIO = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# RATIO = [1]
LAYOUTNUM = 2

class LinearRegressionModel(object):
    
    def __init__(self) -> None:
        self.datalayout_conversion_file = {
            'CHW_2_HWC': 'CHW_2_HWC_15time.csv.json',
            'HWC_2_CHW': 'HWC_2_CHW_15time.csv.json',
            # linear_regression_parameters/collect_data_time/
        }
        self.CHW_file = {
            'depthwise_CHW_conv_k3_s1': 'depthwise_kernel3_stride1_CHW_15time.csv.json',
            'depthwise_CHW_conv_k3_s2': 'depthwise_kernel3_stride2_CHW_15time.csv.json',
            'conv_CHW_N1': 'convolution_kernelN1_CHW_15time.csv.json',
            'conv_CHW_k1': 'convolution_kernel1_stride1_pad0_CHW_15time.csv.json',
            'add': 'add_15time.csv.json',
            'Avg_Pool': 'arm_avepool_q7_HWC_25time.csv.json',
            'Max_Pool_CHW': 'maxpooling_CHW_15time.csv.json',
            'Avg_Pool_CHW': 'maxpooling_CHW_15time.csv.json'
        }
        
        self.HWC_file = {
            'depthwise_conv_HWC': 'depthwise_HWC_15time.csv.json',  
            'conv_HWC_k1': 'convolution_kernel1_stride1_pad0_HWC_15time.csv.json',
            'conv__HWC_N1': 'convolution_kernelN1_HWC_15time.csv.json',
            'add': 'add_15time.csv.json',
            'Avg_Pool': 'arm_avepool_q7_HWC_25time.csv.json',
            'Max_Pool_HWC': 'maxpooling_HWC_15time.csv.json',
            'Avg_Pool_HWC': 'maxpooling_HWC_15time.csv.json'
        }
        self.mod2 = ['conv_CHW_k1']
        self.FLOPs_t = ['depthwise_CHW_conv_k3_s1', 'depthwise_CHW_conv_k3_s2', 'depthwise_conv_HWC'] #基于FLOPs_t
    
    def predict(self, node, name, ratio = 1.0, device = ['m4', 'm7']): #feature, layout, 
        file = 'linear_regression_parameters/collect_data_time/'
        feature = list()
        file_set = list()
        if name in self.datalayout_conversion_file:
            N, OC, OH, OW = list(node.attrs['output_shape'])
            feature.append([OH, OC, OH * OW, OH * OC, OH * OW * OC])
            file_set.append(file + self.datalayout_conversion_file[name])
        else:
            map_file = self.CHW_file if name in self.CHW_file else self.HWC_file
            feature = self.features(node.attrs, name, ratio)
            if isinstance(device, list) or isinstance(device, tuple):
                if len(feature) != 1:
                    file_set.append(file + device[0] + '/' + map_file[name])
                file_set.append(file + device[-1] + '/' + map_file[name])
            else:
                file_set.append(file + device + '/' + map_file[name])
        latency = [0.0, 0.0]    
        for (i, fet, file) in zip(list(range(2)), feature, file_set):
            with open(file, 'r') as json_file:
                model = json.load(json_file)
                for (u, v) in zip(model['coefs'], fet):
                    latency[i] += u * v
                
                latency[i] += model['inter']
                latency[i] = latency[i] * fet[-1] if name in self.FLOPs_t else latency[i]
        # print(latency, feature, model['coefs'])
        # print(name, latency, ratio)
        return max(latency)
    
    def features(self, args, name = str(), ratio = 1.0):
        assert args['op_type'] in ['Conv', 'Depthwise_Conv', 'Avg_Pool', 'Max_Pool', 'add'], f"{args['op_type']} in ['Conv', 'Depthwise_Conv', 'Avg_Pool', 'Max_Pool', 'add']."
        feature = list()
        if args['op_type'] == 'Conv':
            k = args['kernel_size'][0] if isinstance(args['kernel_size'], tuple) or isinstance(args['kernel_size'], list) else args['kernel_size']
            N, IC, IH, IW = list(args['input_shape'])
            N, OC, OH, OW = list(args['output_shape'])
            if name in self.CHW_file:    
                TO = OC
                OC = OC * ratio // 1
                M = OH * OW
                K = k * k * IC
                N = OC // 2 * 2 if name in self.mod2 else OC
                MN = M * N
                MK = M * K
                KN = K * N
                MKN = M * K * N
                if N  != 0:
                    if k == 1:
                        feature.append([M, K, N, MK, KN, MN, MKN])
                    else:
                        # 'KH', 'KW', 'KHKWOH', 'KHKWOW', 'KHKWOC'
                        feature.append([k, k, k * k * OH, k * k * OW, k *k * N, M, K, N, MK, KN, MN, MKN])
                N = TO - N
                MN = M * N
                MK = M * K
                KN = K * N
                MKN = M * K * N
                if N  != 0:
                    if k == 1:
                        feature.append([M, K, N, MK, KN, MN, MKN])
                    else:
                        # 'KH', 'KW', 'KHKWOH', 'KHKWOW', 'KHKWOC'
                        feature.append([k, k, k * k * OH, k * k * OW, k *k * N, M, K, N, MK, KN, MN, MKN])
            else:
                TH = OH 
                OH = OH * ratio // 1
                M = OH * OW
                K = k * k * IC
                N = OC
                MN = M * N
                MK = M * K
                KN = K * N
                MKN = M * K * N
                if OH != 0:
                    feature.append([M, K, N, MK, KN, MN, MKN])
                OH = TH - OH
                M = OH * OW
                K = k * k * IC
                N = OC
                MN = M * N
                MK = M * K
                KN = K * N
                MKN = M * K * N
                if OH != 0:
                    feature.append([M, K, N, MK, KN, MN, MKN])
                        
        elif args['op_type'] == 'Depthwise_Conv':
            k = args['kernel_size'][0] if isinstance(args['kernel_size'], tuple) or isinstance(args['kernel_size'], list) else args['kernel_size']
            N, OC, OH, OW = list(args['output_shape'])
            if name in self.CHW_file:
                # FLOPs = (2 * k * k + 1) * OH * OW * OC
                TO = OC
                OC = OC * ratio // 1
                if OC != 0:
                    feature.append([OH, OW, OC, (2 * k * k + 1) * OH * OW * OC])
                OC = TO - OC
                if OC != 0:
                    feature.append([OH, OW, OC, (2 * k * k + 1) * OH * OW * OC])
            else:
                TH = OH
                OH = OH * ratio // 1
                if OH != 0:
                    feature.append([OH, OW, OC, (2 * k * k + 1) * OH * OW * OC])
                OH = TH - OH 
                if OH != 0:
                    feature.append([OH, OW, OC, (2 * k * k + 1) * OH* OW * OC])
                
        # elif args['op_type'] == 'Avg_Pool':
        #     # ignore ratio
        #     k = args['kernel_size'][0] if isinstance(args['kernel_size'], tuple) or isinstance(args['kernel_size'], list) else args['kernel_size']
            
        #     N, OC, OH, OW = list(args['output_shape'])
        #     # FLOPs = (2 * k * k + 1) * OH * OW * OC
        #     # ['OH', 'OW', 'OC', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC']
        #     # 
        #     TH = (DEVICE * OH + 0.5) // 1
        #     OHOW = TH * OW            
        #     OHOC = TH * OC    
        #     OWOC = OW * OC          
        #     OHOWOC = TH * OW * OC       
        #     # 'OH', 'OW', 'OC', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC'       
        #     feature.append([TH, OW, OC, OHOW, OHOC, OWOC, OHOWOC])
            
        elif args['op_type'] == 'Max_Pool' or args['op_type'] == 'Avg_Pool':
            k = args['kernel_size'][0] if isinstance(args['kernel_size'], tuple) or isinstance(args['kernel_size'], list) else args['kernel_size']
            N, OC, OH, OW = list(args['output_shape'])
            # ['OH', 'OW', 'OC', 'KH', 'KW', 'KHKW', 'OHOW', 'OHOC', 'OWOC', 'OHOWOC', 'KHKWOH', 'KHKWOW', 'KHKWOC']
            if name in self.CHW_file:
                TO = OC
                OC = OC * ratio // 1
                OHOW = OH * OW            
                OHOC = OH * OC    
                OWOC = OW * OC          
                OHOWOC = OH * OW * OC
                if OC != 0:
                    feature.append([OH, OW, OC, k, k, k * k, OHOW, OHOC, OWOC, OHOWOC, k * k* OH, k*k*OW, k*k*OC])
                OC = TO - OC
                OHOW = OH * OW            
                OHOC = OH * OC    
                OWOC = OW * OC          
                OHOWOC = OH * OW * OC
                if OC != 0:
                    feature.append([OH, OW, OC, k, k, k * k, OHOW, OHOC, OWOC, OHOWOC, k * k* OH, k*k*OW, k*k*OC])
            else:
                TH = OH
                OH = OH * ratio // 1
                OHOW = OH * OW            
                OHOC = OH * OC    
                OWOC = OW * OC          
                OHOWOC = OH * OW * OC
                if OH != 0:
                    feature.append([OH, OW, OC, k, k, k * k, OHOW, OHOC, OWOC, OHOWOC, k * k* OH, k*k*OW, k*k*OC])
                OH = TH - OH
                OHOW = OH * OW            
                OHOC = OH * OC    
                OWOC = OW * OC          
                OHOWOC = OH * OW * OC
                if OH != 0:
                    feature.append([OH, OW, OC, k, k, k * k, OHOW, OHOC, OWOC, OHOWOC, k * k* OH, k*k*OW, k*k*OC])
                
        elif args['op_type'] == 'add':
            size = reduce(mul, list(args['input_shape']))
            TSize = size
            size = size * ratio // 1
            if size  != 0:
                feature.append([size])
            size = TSize - size
            if size != 0:
                    feature.append([size])
        return feature
            

class GraphNode(object):
    
    def __init__(self, attrs = None):
        self.in_edges = list()
        self.out_edges = list()
        self.ratio = 1.0
        self.layout = 'HWC' # 0:HWC, 1:CHW
        self.latency = 0.0
        assert attrs
        self.attrs = attrs
        self.layout_strategies = list()
        
    def calcu_latency(self, shape, ratio):
        pass
            
class Graph(object):  
    
    def __init__(self):
        self.layer_map = collections.OrderedDict()
        self.topological_sort = list()
        self.input_name = str()
        self.linear_model = LinearRegressionModel()
        self.eliminated_ratio = list()
        self.eliminated_layout = list()
        self.eliminated_node = None
        
    def node_connection(self):
       for (name, value) in self.layer_map.items():
           for input_name in value.attrs['input_param']:
               self._make_connection(name, input_name)
    
    def init_param(self):
        self.eliminated_ratio = list()
        self.eliminated_layout = list()
        
                   
    def _make_connection(self, name, input_name):
        if input_name != self.input_name:
            if not self.layer_map[input_name] in self.layer_map[name].in_edges:
                self.layer_map[name].in_edges.append(self.layer_map[input_name])
        
            if not self.layer_map[name] in self.layer_map[input_name].out_edges:
                self.layer_map[input_name].out_edges.append(self.layer_map[name])
        
    
    def CreateGraphNode(self, attrs):
        return GraphNode(attrs)
    
    def make_edge_conversion_latency(self):
        for (name, value) in self.layer_map.items():
            output = list()
            for v in value.out_edges:
                # ((0, 0), (0, 1), (1, 0), (1, 1)) 0:HWC, 1:CHW
                # output.append({v.attrs['op_name']: [0, 0, 0, 0]})
                output.append({v.attrs['op_name']: [0, self.linear_model.predict(value, 'HWC_2_CHW') * DEVICE, self.linear_model.predict(value, 'CHW_2_HWC') * DEVICE, 0]})
            value.layout_strategies = output
                
    def __str__(self) -> str:
        code  = str()
        for name, node in self.layer_map.items():
            # print(node.attrs['input_shape'], node.attrs['output_shape'])
            code += 'name: ' + name + '     input_size:' +  str(node.attrs['input_shape'])  + '     ' + '     output_size:' +  str(node.attrs['output_shape'])  + '     ' + 'layout_strategies:' + str([str(v) for v in node.layout_strategies])  + '   input_node:' + str([v.attrs['op_name'] for v in node.in_edges]) + '       output_node: ' + str([v.attrs['op_name'] for v in node.out_edges]) + '\n'
        return code
    

class Paser(object):
    
    def __init__(self, model, shape, concrete_args = None) -> None:
        self.trace = symbolic_trace(model, concrete_args=concrete_args)
        self.modules = dict(self.trace.named_modules())
        # inference output shape
        ShapeProp(self.trace).propagate(torch.rand(shape).to(torch.float32))
        # build graph
        self.graph = Graph()
        # fx graph node 
        self.nodes = list(self.trace.graph.nodes)
      
    def build(self):
        for node in self.nodes:
            args = None
            if node.op == "placeholder":
                input_name = node.name
                # args = {'op_name': node.name,}
                self.graph.input_name = input_name
            elif node.op == "call_function":    
                args = {'op_name': node.name,
                        'op_type': self.get_function_name(node.target),   
                        'input_param': [v.name for v in node.args],
                        'input_shape': node.args[0].meta["tensor_meta"].shape,
                        'output_shape': node.meta["tensor_meta"].shape,
                        }
            elif node.op == "call_module":
                module = self.modules[node.target]
                if isinstance(module, nn.Conv2d):  
                    args = {'op_name': node.name,
                            'op_type': 'Conv',
                            'input_param': [v.name for v in node.args],
                            'input_shape': node.args[0].meta["tensor_meta"].shape,
                            'output_shape': node.meta["tensor_meta"].shape,
                            'kernel_size': module.kernel_size,
                            'stride': module.stride,
                            'padding': module.padding,
                            'dilation': module.dilation,
                            'groups':module.groups,}
                    if args['groups'] != 1:
                        args['op_type'] = 'Depthwise_Conv'
                elif isinstance(module, nn.AvgPool2d):
                    args = {'op_name': node.name,
                            'op_type': 'Avg_Pool',
                            'input_param': [v.name for v in node.args],
                            'input_shape': node.args[0].meta["tensor_meta"].shape,
                            'output_shape': node.meta["tensor_meta"].shape,
                            'kernel_size': module.kernel_size,
                            'stride': module.stride,
                            'padding': module.padding,}
                elif isinstance(module, nn.MaxPool2d):
                    args = {'op_name': node.name,
                            'op_type': 'Max_Pool',
                            'input_param': [v.name for v in node.args],
                            'input_shape': node.args[0].meta["tensor_meta"].shape,
                            'output_shape': node.meta["tensor_meta"].shape,
                            'kernel_size': module.kernel_size,
                            'stride': module.stride,
                            'padding': module.padding,}
            if args != None:
                self.graph.layer_map[args['op_name']] = self.graph.CreateGraphNode(args)
                
        # connection graph edge
        self.graph.node_connection()
        self.graph.make_edge_conversion_latency()
        # print(self.graph)
        
        return self.graph

    def get_function_name(self, node_target):
        import re
        function_name = re.findall(
            r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node_target)
        )[0]
        return function_name 

import copy    

class Scheduler(object):
    
    def __init__(self, model, shape:list) -> None:
        self.graph = Paser(model, shape=shape).build()
        self.source_graph_node_size = len(self.graph.layer_map)
        self.graph_set = [self.graph]
        self.temp_graph = copy.deepcopy(self.graph_set[-1])
        self.schedule = collections.OrderedDict()
        self.run()
        
    def node_elimination(self):
        for name, node in self.temp_graph.layer_map.items():
            if len(node.in_edges) == 1 and len(node.out_edges) == 1:
                return (name, node)
        return (None, None)
    
    def edge_elimination(self, name):
        
        if len(self.temp_graph.layer_map[name].in_edges) == 0 or name == None:
            return False
        
        if self.temp_graph.layer_map[name].out_edges[0] in self.temp_graph.layer_map[name].in_edges[0].out_edges:
            return True
        
        return False
    
    def linear_model_name(self, node, format):
        op_type = node.attrs['op_type']
        # ['Conv', 'Depthwise_Conv', 'Avg_Pool', 'Max_Pool', 'add']
        name = str()
        if format == 0: #HWC
            if op_type == 'Conv':
                k = node.attrs['kernel_size'][0] if isinstance(node.attrs['kernel_size'], tuple) or isinstance(node.attrs['kernel_size'], list) else node.attrs['kernel_size']
                name = 'conv__HWC_N1' if k != 1 else 'conv_HWC_k1'
            elif op_type == 'Depthwise_Conv':
                name = 'depthwise_conv_HWC'
            elif op_type == 'add':
                name = 'add'
            elif op_type == 'Avg_Pool':
                name = 'Avg_Pool_HWC'
            elif op_type == 'Max_Pool':
                name = 'Max_Pool_HWC'
        elif format == 1: #CHW
            if op_type == 'Conv':
                k = node.attrs['kernel_size'][0] if isinstance(node.attrs['kernel_size'], tuple) or isinstance(node.attrs['kernel_size'], list) else node.attrs['kernel_size']
                name = 'conv_CHW_N1' if k != 1 else 'conv_CHW_k1'
            elif op_type == 'Depthwise_Conv':
                s = node.attrs['stride'][0] if isinstance(node.attrs['stride'], tuple) or isinstance(node.attrs['stride'], list) else node.attrs['stride']
                name = 'depthwise_CHW_conv_k3_s1' if s == 1 else 'depthwise_CHW_conv_k3_s2'  
            elif op_type == 'add':
                name = 'add'
            elif op_type == 'Avg_Pool':
                name = 'Avg_Pool_CHW'
            elif op_type == 'Max_Pool':
                name = 'Max_Pool_CHW'
                
        return name
    
    def run(self):
        while(True):
            node_name, node = self.node_elimination()
            if node_name == None:
                break
            edge_eliminated = self.edge_elimination(node_name)
            
            input_node = node.in_edges[0]
            output_node = node.out_edges[0]
            latency_set = [float("inf"), float("inf"), float("inf"), float("inf")]

            strategy_ik = [v for v in input_node.layout_strategies for u, v in v.items() if u == node_name][0]
            strategy_kj = [v for v in node.layout_strategies for u, v in v.items() if u == output_node.attrs['op_name']][0]
            # print(strategy_kj)
            for i in range(LAYOUTNUM):
                for j in range(LAYOUTNUM):
                    latency_k = list()
                    ratio_idex_k = list()
                    for k in range(LAYOUTNUM):
                        latency_r = list()
                        
                        # 对应数据布局的模型名字
                        model_name = self.linear_model_name(node, k)
                        
                        for r in RATIO:    
                            # 不同划分划分比例对应的时延
                            latency_r.append(self.temp_graph.linear_model.predict(node, model_name, r))
                        
                        # 该数据布局对应的一个最优的时延
                        latency_k.append(min(latency_r) + strategy_ik[i * 2 + k] + strategy_kj[k * 2 + j])
                        
                        # 对应数据布局最优的划分比例
                        ratio_idex_k.append(latency_r.index(min(latency_r)))
                    
                    # 不同布局对应的最优的一个边时延
                    latency_set[i * 2 + j] = min(latency_k)
                    idex = latency_k.index(min(latency_k))
                    self.temp_graph.eliminated_layout.append(idex)

                    # 每个边布局对应的一个最优的划分比例
                    self.temp_graph.eliminated_ratio.append(ratio_idex_k[idex]) 
                    # print(self.temp_graph.eliminated_layout, self.temp_graph.eliminated_ratio)      
            eliminate_strategy_node_name = [node_name]
            if edge_eliminated:
                strategy_ij = [v for v in input_node.layout_strategies for u, v in v.items() if u == output_node.attrs['op_name']][0]
                
                latency_set = [i + j for i, j in zip(latency_set, strategy_ij)]
                eliminate_strategy_node_name.append(output_node.attrs['op_name'])
                # input_node.layout_strategies = [{u:v} for v in input_node.layout_strategies for u, v in v.items() if u not in [node_name, output_node.attrs['op_name']]]
                # input_node.layout_strategies.append({output_node.attrs['op_name']: latency_set})
            else:
                input_node.out_edges.append(output_node)
                output_node.in_edges.append(input_node)
             
            input_node.layout_strategies = [{u:v} for v in input_node.layout_strategies for u, v in v.items() if u not in eliminate_strategy_node_name]
            input_node.layout_strategies.append({output_node.attrs['op_name']: latency_set})
            
            input_node.out_edges = [v for v in input_node.out_edges if v != node]
            output_node.in_edges = [v for v in output_node.in_edges if v != node]
            
            self.temp_graph.eliminated_node = (node_name, node)
            self.temp_graph.layer_map.pop(node_name)
            self.graph_set.append(self.temp_graph)
            self.temp_graph = copy.deepcopy(self.graph_set[-1])
            self.temp_graph.init_param()
        # print(self.temp_graph, len(self.graph_set), self.source_graph_node_size)
        
        # print(self.temp_graph.layer_map)
        
        # self.schdule
        name  = list(self.temp_graph.layer_map.keys())[0]
        node = self.temp_graph.layer_map[name]
        output_node = node.out_edges[0]
        latency_i = list()
        ratio_idex_i = list()
        latency_k = list()
        ratio_idex_k = list()
        strategy_ij = [v for v in node.layout_strategies for u, v in v.items() if u == output_node.attrs['op_name']][0]
        for i in range(LAYOUTNUM):
            model_name = self.linear_model_name(node, i)
            latency_r = list()
            for r in RATIO:
                latency_r.append(self.temp_graph.linear_model.predict(node, model_name, r))
            # 最优的划分比例
            latency_i.append(min(latency_r))        
            # 对应数据布局最优的划分比例
            ratio_idex_i.append(latency_r.index(min(latency_r)))
        for k in range(LAYOUTNUM):
            model_name = self.linear_model_name(output_node, k)
            latency_r = list()
            for r in RATIO:
                latency_r.append(self.temp_graph.linear_model.predict(output_node, model_name, r))
            # 最优的划分比例
            latency_k.append(min(latency_r))     
    
            # 对应数据布局最优的划分比例
            ratio_idex_k.append(latency_r.index(min(latency_r)))    

        result = list(range(LAYOUTNUM * LAYOUTNUM))
        
        for i in range(LAYOUTNUM):
            for j in range(LAYOUTNUM):
                result[i * 2 + j] = strategy_ij[i * 2 + j] + latency_i[i] + latency_k[j]
        idex = result.index(min(result))
        print(result, min(result), idex)
        ratio = RATIO
        self.schedule[name] = (idex // LAYOUTNUM, ratio[ratio_idex_i[idex // LAYOUTNUM]] )  #layout format, ratio  
        self.schedule[output_node.attrs['op_name']] = (idex % LAYOUTNUM, ratio[ratio_idex_k[idex % LAYOUTNUM]] )  #layout format, ratio  
        
        for graph in list(reversed(self.graph_set))[0:-1]:
            (node_name, node) = graph.eliminated_node
            node_layout_set = graph.eliminated_layout
            node_ratio_set = graph.eliminated_ratio
            input_node = node.in_edges[0]
            output_node = node.out_edges[0]
            # print(node_name, input_node.attrs['op_name'], output_node.attrs['op_name'])
            (input_layout, input_ratio) = self.schedule[input_node.attrs['op_name']]
            (output_layout, output_ratio) = self.schedule[output_node.attrs['op_name']]
            self.schedule[node_name] = (node_layout_set[input_layout * 2 + output_layout], ratio[node_ratio_set[input_layout * 2 + output_layout]])
            
            
        # print(self.schedule) #self.schedule,
        layout_set = list()
        partitioning_ratio = list()
        for (name, sc) in self.graph_set[0].layer_map.items():
            layout_set.append(self.schedule[name][0])
            partitioning_ratio.append(self.schedule[name][1])
        
        print('operator numbers:', len(layout_set))
        print('data layout:', layout_set)
        print('partitioning strategies:', partitioning_ratio)


import functools

# @functools.lru_cache()
def depth_search(N, layout):
    if N == 0:
        return 0
    conversion_overhed = list(range(4))
    compute_overhead = list(range(22))
    result = list()
    for i in range(2):
        for j in range(11):
            result.append(depth_search(N-1, i) + conversion_overhed[layout * 2 + i] + compute_overhead[i* 11 + j])
    return min(result)

def depth_reolution(N = 7):
    result = list()
    compute_overhead = list(range(22))
    for i in range(2):
        for j in range(11):
            result.append(depth_search(N - 1, i) + compute_overhead[i* 11 + j])
    return min(result) 
                        
if __name__ == '__main__':
    from MCUCNN.MobileNetV2 import MobileNetV2
    from MCUCNN.Proxyless import Proxyless
    from MCUCNN.MnasNet import MnasNet
    from MCUCNN.SmallCifar import SmallCifar
    from timeit import default_timer as timer

    # smallCifar
    model = SmallCifar()
    tic = timer()
    schedule = Scheduler(model, [1, 3, 32, 32])
    toc = timer()
    print('smallCifar schedule time:', toc - tic) # 输出的时间，秒为单位
   
    # MobileNet
    model = MobileNetV2()
    tic = timer()
    schedule = Scheduler(model, [1, 3, 144, 144])
    toc = timer()
    print('MobileNetV2 schedule time:', toc - tic) # 输出的时间，秒为单位
    
    # Proxyless
    model = Proxyless()
    tic = timer()
    schedule = Scheduler(model, shape=[1, 3, 176, 176])
    toc = timer()
    print('Proxyless schedule time:', toc - tic) # 输出的时间，秒为单位
    
    # MnasNet
    model = MnasNet()
    tic = timer()
    schedule = Scheduler(model, shape=[1, 3, 96, 96])
    toc = timer()
    print('MnasNet schedule time:', toc - tic) # 输出的时间，秒为单位
    
    # tic = timer()
    # depth_reolution(7)
    # toc = timer()
    # print('Depth schedule time:', toc - tic) # 输出的时间，秒为单位
    