import torch
import argparse
import os

import fairseq
from fairseq.checkpoint_utils import torch_persistent_save
from fairseq.file_io import PathManager

def state_2_fairseq_model(args,state_file, base_fairseq_model, ignore_embedding ):
    """_summary_

    Args:
        state_file (_type_): fused state dictionary
        base_fairseq_model (_type_): fairseq base model

    Returns:
        _type_: fairseq fused model
    """
    fairseq_model = torch.load(os.path.join(args.checkpoints_dir, base_fairseq_model))
    
    fused_state = torch.load(os.path.join(args.checkpoints_dir,state_file))

    assert len(fairseq_model['model'].keys()) == len(fused_state.keys())
    if not ignore_embedding:
        for key in fairseq_model['model'].keys():
            fairseq_model['model'][key] = fused_state[key]
    else:
        ignore_dict=['encoder.embed_tokens', 'encoder.embed_positions', 'decoder.embed_tokens', 'decoder.embed_positions', 'decoder.output_projection']
        for key in fairseq_model['model'].keys():
            T=0
            for i in ignore_dict:
                if i in key:
                    T=1
            if T==1:
                continue 
            fairseq_model['model'][key] = fused_state[key]
    return fairseq_model

def main(args):
    fairseq_model = state_2_fairseq_model(args, args.fused_state, args.base_model,args.ignore_embedding)
    filename = os.path.join(args.checkpoints_dir, args.fused_state[:-3] + '_fairseq.pt')
    with PathManager.open(filename, "wb") as f:
        torch_persistent_save(fairseq_model, f)
    test = torch.load(filename)
    
    print('>> Convert successfully..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert state_dict to fairsq model.pt')
    parser.add_argument('--checkpoints_dir', default='/home/checkpoints/fusioned_state', type=str, help='checkpoints dir')
    parser.add_argument('--fused_state', default='ende_trimed_0.1_FT_0.9_RI.pt', type=str, help='raw file path')
    parser.add_argument('--base_model', default='base_FT_ende.pt', type=str, help='base model file path')
    parser.add_argument('--ignore_embedding', default=True, type=bool, help='ignore embedding')
    args = parser.parse_args()

    main(args)

    
