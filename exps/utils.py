import copy
from typing import OrderedDict
from cv2 import trace
from numpy import indices
import torch
import ot
from torch._C import device

import routines
import copy
import ground_metric as gm

def update_model(model, update_params):
    state = model.state_dict()
    for k, v in update_params.items():
        state[k] = v
    model.load_state_dict(state)

def get_histogram(args, cardinality, layer_weight = None, 
                    activation=None, dtype=torch.float64):
    if args.importance == 'uniform':
        # print("uniform importance")
        hist =  torch.ones(cardinality, dtype=dtype)/cardinality
    elif args.importance == 'wts':
        assert layer_weight != None, "layer weight can't be none"
        print(f"get importance from weights with method {args.importance_method}")
        hist =  get_importance_hist(layer_weight.reshape(layer_weight.shape[0], -1), args.importance_method, dtype=dtype)
    elif args.importance == 'acts':
        assert activation != None, "activation can't be none"
        print(f"get importance from activations with method {args.importance_method}")
        hist = get_importance_hist(activation, args.importance_method, args.softmax_temperature, dtype=dtype)
    else:
        raise NotImplementedError(f"importance {args.importance} not found") 
    assert hist.max()<1-1e-3, "distribution of importance error"
    return hist.type(args.dtype).detach().cpu()


def regist_hooks(module, parent_name, get_activation_hook, activations, hook_handles):
    for name, submodule in module.named_children():
        name_ = parent_name + '.' + name if parent_name else name
        if hasattr(submodule, 'weight') or hasattr(submodule, 'bias') or len(list(module.children()))==0:
            hook = get_activation_hook(activations, name_)
            hook_handles.append(submodule.register_forward_hook(hook))
            if len(list(module.children())) > 0:
                regist_hooks(submodule, name_, get_activation_hook, activations, hook_handles)
        else:
            regist_hooks(submodule, name_, get_activation_hook, activations, hook_handles)

def print_list_tensor_shape(l):
    for t in l:
        print(t.shape, end=',')
    print()

def align_activation(args, activation0: torch.Tensor, activation1, feature_dim=-1, eps=1e-5):
    def check_dim_identify(tensors, dim=-1):
        a = tensors[0].shape[dim]
        for tensor in tensors:
            if tensor.shape[dim] != a:
                return False
        return True

    try:
        if type(activation0) == list:
            if len(activation0) == 1:
                activation0 = activation0[0]
            else:
                activation0 = torch.cat(activation0, dim=0)
        elif type(activation0) == torch.Tensor:
            pass
        
        if type(activation1) == list:
            if len(activation1) == 1:
                activation1 = activation1[0]
            else:        
                activation1 = torch.cat(activation1, dim=0)
        elif type(activation1) == torch.Tensor:
            pass
    except Exception as e:
        print(e)
        return None, None, None
    M0, dis = p_dis(activation0, activation1, args, feature_dim)    

    M0 = M0 + (torch.eye(M0.size(0)) == 0) * eps
    
    mu_cardinality = activation0.shape[feature_dim]
    nu_cardinality = activation1.shape[feature_dim]
    
    mu = get_histogram(args, mu_cardinality, activation=activation0, dtype=args.dtype)
    nu = get_histogram(args, nu_cardinality, activation=activation1, dtype=args.dtype)
    mu *= mu_cardinality
    nu *= mu_cardinality

    cpuM = M0.data.detach().cpu().type(args.dtype)

    if args.exact:
        # shape of T (mu_cardinality, nu_cardinality)
        T = ot.emd(mu.numpy(), nu.numpy(), cpuM.numpy())
    else:
        T = ot.bregman.sinkhorn(mu.numpy(), nu.numpy(), cpuM, reg=args.reg)

    T = torch.tensor(T)
    ot_cost = torch.multiply(T, cpuM).sum()

    print(f"Ratio of trace to the matrix sum:  {(torch.trace(T)/torch.sum(T)).item():.4f};\t otcost: , {ot_cost.item():.2f};\t distance:{dis}")
    return T, mu, nu

def align_activations(args, activations):
    activations_T = dict(zip(activations[0].keys(), [None]*len(activations[0].keys())))
    for name in activations[0].keys():
        print(f">>> {name}")
        is_conv = (activations[0][name]['input'].ndim == 4)
        if is_conv:
            feature_dim = 1
        else:
            feature_dim = -1
        T_var_in, _, _ = align_activation(args, activations[0][name]['input'], activations[1][name]['input'], feature_dim)
        T_var_out, _, _ = align_activation(args, activations[0][name]['output'], activations[1][name]['output'], feature_dim)

        activations_T[name] = {'input': T_var_in, 'output': T_var_out} 
    return activations_T

def rstrip(s, tails):
    for tail in tails:
        if s.endswith(tail):
            s = s[0: len(s)-len(tail)]
    return s

import activation_utils as au
def adaptive_fuse_model2(args, model, model1, activation_manager, inplace=False, recompute_acts=True, skip_in=[], skip_out=[], p0=0.5):

    model0 = model if inplace else copy.deepcopy(model)
    model1_initial = copy.deepcopy(model1)
    am: au.ActivationsManager = activation_manager
    am.reset()

    aligned_params = {}
    avg_aligned_params = {}

    actual_activations_T = {rstrip(k, [".weight", ".bias", ".q_bias", ".v_bias"]): {} for k,v in model0.named_parameters()}

    last_module_name = ""
    with torch.no_grad():
        for (layer0_name, layer0_weight), (layer1_name, layer1_weight) in zip(model0.named_parameters(), model1.named_parameters()):
            layer0_weight_shape = layer0_weight.shape
            name = rstrip(layer0_name, [".weight", ".bias", ".q_bias", ".v_bias"])

            is_conv = len(layer0_weight.shape) == 4
            print(f"\n\n>>> layer name: {name} ||| param name: {layer0_name}")
            print("layer shape:", layer0_weight.shape, "activations shape")
            if name == "encoder.embed_tokens" or "embeddings" in name:
                feature_dim = -1
            else:
                feature_dim = 0

            is_conv = (layer0_weight.ndim >= 3)
            if is_conv:
                print("is_conv")
                act_feature_dim = 1
            else:
                act_feature_dim = -1

            T_var_in = None 
            T_var_out = None 
            if name not in skip_in and layer0_weight.ndim != 1: 
                if "input" in actual_activations_T[name].keys(): 
                    T_var_in = actual_activations_T[name]['input'] 
                else: 
                    T_var_in, _, _ = align_activation(args, am.origin_activations[0][name]['input'], am.activations[0][name]['input'], feature_dim=act_feature_dim) # ??
                    actual_activations_T[name]['input'] = T_var_in 
            if name not in skip_out: 
                if "output" in actual_activations_T[name].keys(): 
                    T_var_out = actual_activations_T[name]['output'] 
                else: 
                    T_var_out, _, _ = align_activation(args, am.activations[0][name]['output'], am.activations[1][name]['output'], feature_dim=act_feature_dim) 
                    actual_activations_T[name]['output'] = T_var_out 

            layer0_aliged = layer0_weight.clone()
            if T_var_in != None and len(layer0_aliged.shape)>1:
                print("[align in]")
                if is_conv:
                    print("zzzz conv: ", layer0_weight.shape, T_var_in.shape)
                    layer0_aliged = torch.einsum("oiwh,ix->oxwh",layer0_aliged.type(args.dtype), T_var_in)
                else:
                    if feature_dim == 0:
                        layer0_aliged = layer0_aliged.type(args.dtype) @ T_var_in
                    elif feature_dim == -1:
                        layer0_aliged = T_var_in.t() @ layer0_aliged.type(args.dtype) 


            if T_var_out != None:
                print("[align out]")
                if feature_dim == 0:
                    layer0_aliged = (T_var_out.t() @ layer0_aliged.reshape(layer0_weight_shape[0], -1).type(args.dtype)).float()
                elif feature_dim == -1:
                    layer0_aliged = (layer0_aliged.reshape(-1, layer0_weight_shape[-1]).type(args.dtype) @ T_var_out).float()
                else:
                    raise NotImplementedError("feature_dim only suppor 0 or -1 now")
            layer0_aliged = layer0_aliged.view(layer0_weight_shape)
            if not layer0_name.split('.')[1] in ['embed_tokens', 'embed_positions']:
                avg_aligned_params[layer0_name] = layer0_aliged * p0 + layer1_weight * ( 1 - p0 )
            aligned_params[layer0_name] = layer0_aliged
        
            if recompute_acts and p_dis(layer0_aliged, layer0_weight)>1e-6: 
                update_model(model0, aligned_params)   
                am.update_model(0, layer0_name, layer0_aliged)
                print('[recompute]')
                am.recompute(last_module_name, name)
                last_module_name = name
                in_dis = -1
                _, out_dis = p_dis(am.activations[0][name]['output'], am.activations[1][name]['output'], args)
                print("distance after recompute: ",  out_dis)
            else:
                recomputed_acts = False
    avged_model = copy.deepcopy(model1_initial)
    update_model(avged_model, avg_aligned_params)

    return avged_model, model0, actual_activations_T, am

def activations_sorts(activation, feature_dim=0):
    axis = list(range(activation.ndim))
    if feature_dim < 0:
        feature_dim = activation.ndim + feature_dim
    axis.remove(feature_dim)
    activation = activation.std(axis=axis)

    values, indices = torch.sort(activation)
    return indices

def args_modify(args, act_num_samples=1000,
                geom_ensemble_type='acts',
                importance="uniform",
                importance_method="softmax",
                softmax_temperature="10",
                dtype=torch.float64,
                mem_efficient = False,
                exact=True,
                reg = 0.1
                ):
    args.act_num_samples = act_num_samples 

    args.geom_ensemble_type = geom_ensemble_type

    args.importance = importance
    args.importance_method = importance_method
    args.softmax_temperature = softmax_temperature
    args.dtype = dtype

    args.mem_efficient = mem_efficient

    args.exact = exact
    args.reg = reg

def process_activations_T(activations_T, mode="None"):
    """
    args:
        mode: in | out | None
    """
    processed_a_T = copy.deepcopy(activations_T)
    before_k = None
    for k, v in processed_a_T.items():
        if before_k != None:
            if mode == "in":
                v['input'] = processed_a_T[before_k]['output']
            elif mode == "out":
                processed_a_T[before_k]['output'] = v['input']
        before_k = k
    return processed_a_T

def save_fused_model(fused_model, path, model0_path=None):
    checkpoint = {}
    if model0_path != None:
        checkpoint = torch.load(model0_path)
    
    checkpoint['model'] = fused_model.state_dict()
    torch.save(checkpoint, path)

def tune_activations_T_for_transformers(activations_T_x):
    activations_T = copy.deepcopy(activations_T_x)
    for layer_idx in range(6):
        if layer_idx==0:
            prev_last_module = 'embed_tokens'
        else:
            prev_last_module = f'layers.{layer_idx-1}.final_layer_norm'

        activations_T[f'encoder.layers.{layer_idx}.self_attn.q_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']
        activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']
        activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['input'] = activations_T[f'encoder.{prev_last_module}']['output']

        activations_T[f'decoder.layers.{layer_idx}.self_attn.q_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']
        activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']
        activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['input'] = activations_T[f'decoder.{prev_last_module}']['output']

        activations_T[f'encoder.layers.{layer_idx}.self_attn.q_proj']['output'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn.q_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.q_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.k_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['output'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.k_proj']['output']  # 感觉不需要
        activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.k_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.v_proj']['output'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.k_proj']['output'] 
        
        activations_T[f'encoder.layers.{layer_idx}.self_attn.out_proj']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.v_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn.out_proj']['input'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.v_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn.out_proj']['input'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn.v_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.self_attn_layer_norm']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn.out_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.self_attn_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.self_attn.out_proj']['output'] 
        activations_T[f'decoder.layers.{layer_idx}.encoder_attn_layer_norm']['input'] - activations_T[f'decoder.layers.{layer_idx}.encoder_attn.out_proj']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.fc1']['input'] = activations_T[f'encoder.layers.{layer_idx}.self_attn_layer_norm']['output']  # 本来就差不多
        activations_T[f'decoder.layers.{layer_idx}.fc1']['input'] = activations_T[f'decoder.layers.{layer_idx}.encoder_attn_layer_norm']['output'] 

        activations_T[f'encoder.layers.{layer_idx}.fc2']['input'] = activations_T[f'encoder.layers.{layer_idx}.fc1']['output']
        activations_T[f'decoder.layers.{layer_idx}.fc2']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc1']['output']

        activations_T[f'encoder.layers.{layer_idx}.final_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc2']['output']
        activations_T[f'decoder.layers.{layer_idx}.final_layer_norm']['input'] = activations_T[f'decoder.layers.{layer_idx}.fc2']['output']

    activations_T['decoder.output_projection']['input'] = activations_T[f'decoder.layers.5.final_layer_norm']['output']
    return activations_T


def check_activations_T(activations_T):
    key = "encoder.embed_tokens"
    print("non diag sum:")
    for key in activations_T.keys():
        try:
            input_diag = activations_T[key]['input'].diag().sum()
            output_diag = activations_T[key]['output'].diag().sum()
            input_non_diag = activations_T[key]['input'].size(0) - input_diag
            output_non_diag = activations_T[key]['output'].size(0) - output_diag
            print(f"{key}: {input_non_diag}, {output_non_diag}")
        except Exception as e:
            print(key, e)

def check_activations(activations):
    """
    compare the difference of two models' activations
    """
    print("input_diff, output_diff, input_sum_diff, output_sum_diff")
    for key in activations[0].keys():
        try:
            print(f"{key}: input shape, {activations[0][key]['input'][0].shape}, output shape{activations[0][key]['output'][0].shape}")
        except Exception as e:
            print("error", key, e)

def check_model_parameters(model, activation_manager):
    for name, p in model.named_parameters():
        m_name = rstrip(name, ['.weight', '.bias'])
        print(f"\n>>> {m_name}")
        try:
            print(activation_manager.activations[0][m_name]['input'].shape, activation_manager.activations[0][m_name]['output'].shape)
        except Exception as e:
            print(e)
        
def dis(t1, t2, tag="", method="diff"):
    def _pairwise_distances(x, y):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist = torch.clamp(dist, min=0.0)
        dist = dist ** (1/2)
        return dist
    if method == 'diff':
        print(tag, (t1-t2).abs().sum()/(t1.abs().sum()+t2.abs().sum()))
    else:
        return _pairwise_distances(t1, t2)

def model_param_diff(net1, net2, thred=0.02):
    for (name, p1), (_, p2) in zip(net1.named_parameters(), net2.named_parameters()):
        d = (p1-p2).abs().sum()/(p1.abs()+p2.abs()).sum()
        if d>thred:
            print(name, d, p1.shape)
def activation_diff(activationManager):
    for name in activationManager.origin_activations[0].keys():
        d1, d2 = -1, -1
        if type(activationManager.origin_activations[0][name]['input']) == torch.Tensor:
            op1 = activationManager.origin_activations[0][name]['input']
            op2 = activationManager.origin_activations[1][name]['input']
            d1 = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
        if 'output' in activationManager.origin_activations[0][name].keys() and type(activationManager.origin_activations[0][name]['output']) == torch.Tensor:
            op1 = activationManager.origin_activations[0][name]['output']
            op2 = activationManager.origin_activations[1][name]['output']
            d2 = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
        print(name, d1, d2)
def p_dis(op1, op2, args=None, feature_dim=-1):
    if args!=None:
        activation0, activation1 = op1, op2
        if feature_dim == 1:
            activation0 = torch.einsum('bf...->fb...', activation0)
            activation1 = torch.einsum('bf...->fb...', activation1)

        elif feature_dim == -1:
            activation0 = activation0.reshape(-1, activation0.shape[-1]).T
            activation1 = activation1.reshape(-1, activation1.shape[-1]).T
        activation0 = activation0.reshape(activation0.shape[0], -1).contiguous()
        activation1 = activation1.reshape(activation1.shape[0], -1).contiguous()
        M0 = gm.GroundMetric.PROCESS(args, activation0, activation1)    
        d = torch.trace(M0)
        return M0, d
    else:
        d = (op1-op2).abs().sum()/(op1.abs().sum()+op2.abs().sum())
    return d
