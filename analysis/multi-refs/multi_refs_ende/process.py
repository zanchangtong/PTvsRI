


with open('/workspace/data/users/zanchangtong1/data/multi_refs_ende/wmt14-en-de.extra_refs', 'r', encoding='utf-8') as fin:
    all_data={}
    
    for line in fin:
        type, line = line.strip().split('\t')
        type = type.split('-')[0]
        try:
            all_data[type].append(line + '\n')
        except:
            all_data[type] = [line + '\n']


for type in all_data.keys():
    path='/workspace/data/users/zanchangtong1/data/multi_refs_ende/'+type
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(all_data[type])
        f.flush()
