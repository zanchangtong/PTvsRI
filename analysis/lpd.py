import numpy as np
import torch
from fairseq import utils
from fairseq import checkpoint_utils, options, progress_bar, utils
from scipy.stats import entropy
import matplotlib.pyplot as plt

def Bucket_Sort(array, bucketsize):
        minValue = min(array)
        maxValue = max(array)
        bucketcount = (maxValue + 1) // bucketsize
        bucket_lists = [[] for i in range(int(bucketcount) + 1)]
        
        for i in array:
            bucket_index = i // bucketsize
            bucket_lists[int(bucket_index)].append(i)
        return bucket_lists
    
def compute_entropy(probs):
    return entropy(probs, base=2)

def eval_negative_diversity(model, test_batch_iterator):
    entropy_on_test_detailed=[]
    loss = 0
    sample_sum = 0
    model.eval()
    test_iter = test_batch_iterator.next_epoch_itr(shuffle=True)
    soft_max = torch.nn.Softmax(dim=2)
    with torch.no_grad():
        for idx, sample in enumerate(test_iter):
            sample = utils.move_to_cuda(sample)
            output = model(**sample['net_input'])

            tgt_size = ~ sample['target'].eq(1) 
            tgt_size = tgt_size.sum(1) - 2 
            output = soft_max(output[0]).cpu().detach().numpy()
            tmp = []
            for idx in range(np.shape(output)[0]):
                for idx_2 in range(np.shape(output)[1]):
                    p = output[idx][idx_2]
                    entro = compute_entropy(p)
                    tmp.append(entro)
                    if idx_2 == tgt_size[idx]:
                        break
                entropy_on_test_detailed.append(tmp)
                tmp = []
    return {k: v for k, v in enumerate(entropy_on_test_detailed)}

if __name__=='__main__':
    RI_model_path ='ende_RI_trimed_dict.pt'
    PT_model_path="ende_PT_trimed_dict.pt"

    bucket_size = []
    for model_path in [RI_model_path, PT_model_path]:
        overrides = None
        model, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            arg_overrides=overrides,
            )
        model = model[0].cuda()

        subset = 'test'
        task.load_dataset(subset, combine=False, epoch=0)
        dataset = task.dataset(subset)

        batch_itr = task.get_batch_iterator(
                    dataset=dataset,
                    max_tokens=2048,
                    max_positions=utils.resolve_max_positions(
                            task.max_positions(),
                            *[m.max_positions() for m in [model]],
                            ),
                            ignore_invalid_inputs=True,
                            )

        E = eval_negative_diversity(model, batch_itr)
        fltten_entropy=[]
        for key, value in E.items():
            fltten_entropy += value
        fltten_entropy = np.array(fltten_entropy)

        buckets = Bucket_Sort(fltten_entropy, 2)
        bucket_size.append([len(b) for b in buckets])
    

    print('wmt19ende.RI.lpd value: \n {}'.format(bucket_size[0]))
    print('wmt19ende.PT.lpd value: \n {}'.format(bucket_size[1]))
