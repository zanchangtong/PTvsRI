import sys
import utils
import activation_utils as au
import prepare
import copy
import fairseq
fairseq.modules.MultiheadAttention.functional_self_attention = False
import torch


if __name__ == '__main__':
    exp=sys.argv[1]
    aux_task = sys.argv[2] 
    tgt_task = sys.argv[3] 
    pend = sys.argv[4]
    p0 = sys.argv[5]
    
    tag = 'mBART'
    args, net1, net2, aux_inputs, tgt_inputs = prepare.mbart_translation_test(exp=exp, aux_task=aux_task, tgt_task=tgt_task) # prepare model and input. Need to modify model path
    
    acts_inputs = aux_inputs
    v2_activationManager = au.ActivationsManager(args, [copy.deepcopy(net1), copy.deepcopy(net2)], acts_inputs)

    args.mem_efficient = True

    v2_avged_model, v2_aligned_model0, v2_actual_activations_T, v2_am = utils.adaptive_fuse_model2(args, net1, net2, v2_activationManager,
        skip_in=['encoder.embed_tokens', 'encoder.embed_positions', 'decoder.embed_tokens', 'decoder.embed_positions', 'decoder.layernorm_embedding', 'encoder.layernorm_embedding'], \
            skip_out=['encoder.embed_tokens', 'encoder.embed_positions', 'decoder.embed_tokens', 'decoder.embed_positions', 'decoder.output_projection'], p0=float(p0))
    torch.save( v2_avged_model.state_dict(), \
        '/home/checkpoints/fusioned_state/{}_FT_main.pt'.format(pend))

