import copy
from curses.ascii import isdigit
from operator import sub
from typing import Dict, OrderedDict
from numpy import indices
import torch
import ot
import torch.nn.functional as F
from torch._C import device

import utils
import routines
import copy
from torch import nn


def save_activations_(model_name, activation, dump_path):
    mkdir(dump_path)
    pickle_obj(
        activation,
        os.path.join(dump_path, 'model_{}_activations'.format(model_name))
    )

def rstrip(s, tail):
    if s.endswith(tail):
        return s[0: len(s)-len(tail)]
    return s

def print_list_tensor_shape(l):
    for t in l:
        print(t.shape, end=',')
    print()
def get_activation_hook(activation, name):
    def hook(model, inputs, output):
        # if name not in activation:
        #     activation[name] = {'input': [], 'output': []}
        if (type(inputs) == tuple or type(inputs) == list) and len(inputs)>0:
            # inputs = [inp.detach() if type(inp)==torch.Tensor else inp for inp in inputs ]
            inputs = inputs[0].detach()
        # if type(inputs) == tuple:
        #     inputs = [inp.detach() if type(inp)==torch.Tensor else inp for inp in inputs ]
        if type(output) == torch.Tensor:
            output = output.detach()
        # assert len(inputs) == 1
        activation[name] = {}
        # print(name, "hook")
        activation[name]['input'] = inputs  #.append(inputs[0])
        activation[name]['output'] = output #.detach() #.append(output.detach())
    return hook



def regist_hooks(module, parent_name, get_hook, activations, hook_handles):
    for name, submodule in module.named_children():
        name_ = parent_name + '.' + name if parent_name else name
        # print(name_)
        # if hasattr(submodule, 'weight') or hasattr(submodule, 'bias') or len(list(module.children()))==0:
            # print("set forward hook for module named: ", name_)
            # activations[name_] = {'input': [], 'output': []}
        hook = get_hook(activations, name_)
        hook_handles.append(submodule.register_forward_hook(hook))
        regist_hooks(submodule, name_, get_hook, activations, hook_handles)
    # for name, submodule in module.named_modules():
    #     hook = get_activation_hook(activations, name)
    #     hook_handles.append(submodule.register_forward_hook(hook))

def get_backward_hook(gradients, name):
    def hook(module, grad_in, grad_out):
        if name not in gradients:
            gradients[name] = []

        if type(grad_out) == tuple:
            if len(grad_out) != 1:
                print(name, len(grad_out))
            grad_out = grad_out[0]
            # assert len(grad_out) == 1

        if isinstance(grad_out, torch.Tensor):
            grad_out = grad_out.detach()
        gradients[name].append(grad_out)
    return hook

def concat(tensor_list):
    value = tensor_list
    if type(tensor_list[0]) == torch.Tensor:
        value = torch.cat(tensor_list, dim=0)
    else:
        print(f"warning: {type(tensor_list[0])} can't be concat")
    return value

class GredientsManager:
    def __init__(self, args, models, inputs) -> None:
        self.args = args
        self.models = models #copy.deepcopy(models)
        self.inputs = inputs

        torch.manual_seed(args.activation_seed)
        self.activation_gradients = {} 
        self.hook_handles = []

        self.regist_hooks()
        self.forward()
        self.remove_hooks()

        self.fisher_infos = {}

        pass
    
    def regist_hooks(self):
        for idx, model in enumerate(self.models):
            self.activation_gradients[idx] = {}
            # regist_backward_hooks(model, "", get_backward_hook, activation_gradients[idx], hook_handles)
            regist_hooks(model, "", get_backward_hook, self.activation_gradients[idx], self.hook_handles)
    
    def forward(self):
        for model in self.models:
            model.train()
            if not type(self.inputs) == list:
                inputs = [self.inputs]
            for i, data in enumerate(inputs):
                if type(data) == tuple:
                    input, target = data
                    # print(input.shape, target)
                    output = model(input)
                    # loss = criterion(output, target)
                    loss = F.nll_loss(output, target, size_average=False)
                elif type(data) == dict:
                    outputs = model(**data)
                    loss = outputs["loss"]

                # print(output.argmax(), target)
                loss.backward()
        for idx in self.activation_gradients.keys():
            for name in self.activation_gradients[idx].keys():
                self.activation_gradients[idx][name] = concat(self.activation_gradients[idx][name])
                
    def get_fisher_infos(self, element_or_neuron="element", grad_type="square", activation_feature_dim=1):
        if grad_type == "absolute":
            grad_method = torch.abs
        elif grad_type == "square":
            grad_method = torch.square

        for idx, model in enumerate(self.models):
            self.fisher_infos[idx] = {}
            if element_or_neuron == "neuron":
                for pname, _ in model.named_parameters():
                    # activation_gradients[idx][name] = torch.cat(activation_gradients[idx][name], dim=0)            
                    # print(activation_gradients[idx][f"{name}"].shape)
                    
                    # axis = [0] + list(range(2, len(self.activation_gradients[idx][name].shape)))
                    name = rstrip(pname, '.weight')
                    name = rstrip(name, '.bias')

                    axis = list(range(len(self.activation_gradients[idx][name].shape)))
                    del axis[activation_feature_dim]
                    # print(axis)
                    self.fisher_infos[idx][pname] = grad_method(self.activation_gradients[idx][name]).sum(axis=axis)

            elif element_or_neuron == "element":
                for pname, p in model.named_parameters():
                    self.fisher_infos[idx][pname] = grad_method(p.grad).data
                    # print(p.grad)
                    # break
                    # p.grad.zero_()
        return self.fisher_infos


    def remove_hooks(self):
        for hook_handle in self.hook_handles:
            # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
            hook_handle.remove()

class ActivationsManager: 
    def __init__(self, args, models, inputs) -> None: 
        self.args = args 
        self.models = models #copy.deepcopy(models) 
        self.model_states = [model.state_dict() for model in models] 
        self.inputs = inputs 

        torch.manual_seed(args.activation_seed) 
        self.activations = {} 
        self.hook_handles = [] 
 
        # assert args.disable_bias

        # handle below for bias later on!
        # Set forward hooks for all layers inside a model
        for idx, model in enumerate(self.models):
            self.regist_hooks(idx)
            # self.activations[idx] = OrderedDict()
            # regist_hooks(model, "", get_activation_hook, self.activations[idx], self.hook_handles)
        if isinstance(self.inputs, dict):
            self.single_dict = True
        else:
            self.single_dict = False
            
        if self.single_dict:
            data = self.preprocess_inputs(inputs)
            with torch.no_grad():
                if args.gpu_id != -1:
                    data = data[0].cuda(args.gpu_id)
                for idx, model in enumerate(self.models):
                    model.eval()
                    self.compute(model, idx, data)
        else:
            data = [self.preprocess_inputs(inputs[0]), self.preprocess_inputs(inputs[1])]
            with torch.no_grad():
                if args.gpu_id != -1:
                    data = [data[0].cuda(args.gpu_id), data[1].cuda(args.gpu_id)]
                for idx, model in enumerate(self.models):
                    model.eval()
                    self.compute(model, idx, data[idx])

        self.origin_activations = copy.deepcopy(self.activations)

        # Remove the hooks (as this was intefering with prediction ensembling)
    def regist_hooks(self, idx):
        self.activations[idx] = {}
        regist_hooks(self.models[idx], "", get_activation_hook, self.activations[idx], self.hook_handles)
    
    def remove_hooks(self):
        for hook_handle in self.hook_handles:
            # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
            hook_handle.remove()

    def save_activations(self, dump_path):
        # Combine the activations generated across the number of samples to form importance scores
        # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'
        for idx, model in enumerate(self.models):
            # normalize_activations(args, activations[idx], args.activation_mode)   
            # Dump the activations for all models onto disk
            save_activations_(idx, self.activations[idx], dump_path)

    def compute(self, module, idx,  data):
        if type(data) == tuple: # input, target
            self.activations[idx][""] = {"input": data[0]}
            module(data[0])
        elif type(data) == dict:
            self.activations[idx][""] = {"input": data}
            module(**data) 
        else: 
            self.activations[idx][""] = {"input": data}
            module(data)
    def update_model(self, idx, name, p):
        self.model_states[idx][name] = p
        self.models[idx].load_state_dict(self.model_states[idx])
        # utils.update_model(self.models[idx], {name: p})

    def preprocess_inputs(self, acts_inputs):
        if type(acts_inputs) == list:
            if type(acts_inputs[0]) == tuple:
                assert len(acts_inputs[0]) == 2
                data = torch.cat([it[0] for it in acts_inputs], dim=0)
                target = torch.cat([it[1] for it in acts_inputs], dim=0)
            elif type(acts_inputs[0]) == dict:
                data = {}
                for k in acts_inputs[0].keys():
                    data[k] = torch.cat([it[k] for it in acts_inputs], dim=0)
            return data
        return acts_inputs
    def longestCommonPrefix(self, *strs):
        return ''
        if len(strs)==0:
            return ""
        s1 = min(strs).split(".")
        s2 = max(strs).split(".")
        common = s1 if len(s1) <= len(s2) else s2

        for i in range(min(len(s1), len(s2))):
            if s1[i] != s2[i]:
                common = common[:i]
                break
        while len(common)>0:
            try:
                tmp = self.activations[0][".".join(common)]
                break
            except KeyError:
                common = common[:-1]
        return ".".join(common)
    
    def recompute(self, before_module_name="", last_module_name="", idx=0):
        self.remove_hooks()
        self.regist_hooks(0)
        parent_module_name = self.longestCommonPrefix(before_module_name, last_module_name).rstrip('.')
        print("parent_module_name: ", parent_module_name)
        
        model = self.models[idx]
        with torch.no_grad():
            if parent_module_name == '' and self.single_dict:
                data = self.inputs
            elif parent_module_name == '':
                data = self.inputs[idx]
            else:
                data = self.activations[idx][parent_module_name]['input']
            if self.args.gpu_id != -1:
                data = data.cuda(self.args.gpu_id)
            model.eval()
            module = model.get_submodule(parent_module_name)
            
            # print(module)
            if type(data) == tuple or type(data )== list: # input, target
                module(data[0])
            elif type(data) == dict:
                module(**data)
            else:
                module(data) 
    
    def reset(self):
        self.activations = copy.deepcopy(self.origin_activations)

    def __del__(self):
        for hook_handle in self.hook_handles:
            # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
            hook_handle.remove()
            pass

def list_to_tensor(activations):
    for idx in activations.keys():
        for name in activations[idx].keys():
            for inout in activations[idx][name].keys():
                if type(activations[idx][name][inout]) == list and type(activations[idx][name][inout][0]) == torch.Tensor:
                    activations[idx][name][inout] = torch.cat(activations[idx][name][inout], dim=0)
            
def compute_activations_across_models_zz(args, models, inputs, activations=None):
    def get_activation_hook(activation, name):
        def hook(model, inputs, output):
            if name not in activation:
                activation[name] = {'input': [], 'output': []}
            if type(inputs)== tuple:
                inputs = [inp.detach() for inp in inputs]
            
            assert len(inputs) == 1
            activation[name]['input'].append(inputs[0])
            activation[name]['output'].append(output.detach())

        return hook

    torch.manual_seed(args.activation_seed)
    
    activations = {} if activations == None else activations

    hook_handles = []
    assert args.disable_bias

    # handle below for bias later on!
    # Set forward hooks for all layers inside a model
    for idx, model in enumerate(models):
        activations[idx] = OrderedDict()
        utils.regist_hooks(model, "", get_activation_hook, activations[idx], hook_handles)
    # print(activations)
    # exit()
    for data in inputs:
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        for idx, model in enumerate(models):
            if type(data) == tuple: # input, target
                model(data[0])
            elif type(data) == dict:
                model(**data) 


    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    # for idx, model in enumerate(models):
    #     normalize_activations(args, activations[idx], args.activation_mode)   
    #     # Dump the activations for all models onto disk
    #     if dump_activations and dump_path is not None:
    #         utils.save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for hook_handle in hook_handles:
        # print(type(hook_handle))  # <class 'torch.utils.hooks.RemovableHandle'>
        hook_handle.remove()
    return activations

