import argparse
import spacy_udpipe
import multiprocessing


parser = argparse.ArgumentParser(description="Get sentence-level LV and LD")

parser.add_argument('--input_path', type=str, help='the file need to be processed')
parser.add_argument('--output_LV', type=str, help='the file of sentence-level LV')
parser.add_argument('--output_LD', type=str, help='the file of sentence-level LD')
parser.add_argument('--output_L', type=str, help='the file of length')
parser.add_argument('--output', type=str, help='the file of corpus_level LD, LV')
parser.add_argument('--num_workers', type=int, help='number of workers')
parser.add_argument('--lang', type=str, help='language of input')
parser.add_argument('--model', type=str, help='the model to tag pos')


import time

start = time.time()


def analysis_data(input_file, output_LV, output_LD, output_L, output, num_workers, model, lang):
    chars_number = []
    unique_words = dict()
    words_number = []
    context_words_number = []

    unique_words_total = {}
    unique_words = []

    context_words = ["ADJ", "ADV", "VERB", "NOUN"]
    sentence_level_LDs = []
    sentence_level_LVs = []

    part_level_LDs = []
    part_level_LVs = []

    #nlp = spacy_udpipe.load_from_path(lang=lang, path=model, meta={"description": "Custom model"})

    #LV_sentence_level = []
    #LD_sentence_level = []
    #print("input_path:", input_file)

    with open(input_file, 'r') as f1:
        lines = f1.readlines()

    # multiprocessing
    pool = multiprocessing.Pool(processes=num_workers)

    final_results = []

    num_data = len(lines) // num_workers + 1

    for i in range(num_workers):
        res = pool.apply_async(treat, (lines[i * num_data: (i+1) * num_data], model, context_words, lang, i, ))
        final_results.append(res)


    part_ids = []
    sentence_Ls = []

    for res in final_results:
        results = res.get()
        sentence_level = results["sentence_level"]
        
        sentence_level_LVs.append(sentence_level[0])
        sentence_level_LDs.append(sentence_level[1])
        sentence_Ls.append(sentence_level[2])

        part_level = results["part_corpus"]
        part_level_LVs.append(part_level[0])
        part_level_LDs.append(part_level[1])
        part_ids.append(part_level[2])

        corpus = results["total"]
        chars_number.append(corpus[0])
        words_number.append(corpus[1])
        context_words_number.append(corpus[2])
        unique_words.append(corpus[3])

    for unique_word in unique_words:
        for key in unique_word.keys():
             if key not in unique_words_total:
                 unique_words_total[key] = len(unique_words_total)

    sentence_LV_result = []
    sentence_LD_result = []
    sentence_L_result = []

    with open(output_LV, 'w') as f1:
        for sentence_LV in sentence_level_LVs:
            for i in sentence_LV:
                f1.write(str(i))
                f1.write("\n")
                sentence_LV_result.append(i)

    with open(output_LD, 'w') as f2:
        for sentence_LD in sentence_level_LDs:
            for i in sentence_LD:
                f2.write(str(i))
                f2.write("\n")
                sentence_LD_result.append(i)

    with open(output_L, 'w') as f4:
        for sentence_L in sentence_Ls:
            for i in sentence_L:
                f4.write(str(i))
                f4.write("\n")
                sentence_L_result.append(i)


    with open(output, 'w') as f3:
        f3.write("part of corpus:\n")
        for num in range(len(part_ids)):
            f3.write("id:{} LV:{} LD:{}\n".format(part_ids[num], part_level_LVs[num], part_level_LDs[num]))
        
        f3.write("sentence-level:\n")
        f3.write("sentence_level LV:{} sentence-level LD:{}\n".format(sum(sentence_LV_result)/len(sentence_LV_result), sum(sentence_LD_result)/len(sentence_LD_result)))

        f3.write("total corpus:\n")
        f3.write("LV:{} LD:{} avg_L:{}\n".format(len(unique_words_total) / sum(words_number), sum(context_words_number) / sum(words_number), sum(sentence_L_result) / len(sentence_L_result)))


def treat(lines, model, context_words, lang, num):
    nlp = spacy_udpipe.load_from_path(lang=lang,path=model, meta={"description": "Custom 'en' model"})
    sentence_level_LD = []
    sentence_level_LV = []
    sentence_L = []

    word_number = []
    char_number = []
    context_word_number = []

    unique_words = dict()
    print("length:", len(lines))

    for i in range(len(lines)):
        unique_word = dict()
        line = lines[i]

        if i % 5000 == 0:
            print(i)

        doc=nlp(line)
        char_number.append(len(line))
        word_number.append(len(doc))
        sentence_L.append(len(doc))
        con_number = 0

        for word in doc:
            if word.text not in unique_word:
                unique_word[word.text] = len(unique_word)
            if word.text not in unique_words:
                unique_words[word.text] = len(unique_words)
            if word.pos_ in context_words:
                con_number += 1

        context_word_number.append(con_number)
        sentence_level_LV.append(len(unique_word) / len(doc))
        sentence_level_LD.append(con_number / len(doc))

        part_LV = len(unique_words) / sum(word_number)
        part_LD = sum(context_word_number) / sum(word_number)

    return {"sentence_level":[sentence_level_LV, sentence_level_LD, sentence_L], 'part_corpus': [part_LV, part_LD, num], 'total': [sum(char_number), sum(word_number), sum(context_word_number), unique_words]}


args = parser.parse_args()
print(args)
input_file = args.input_path
output_LV = args.output_LV
output_LD = args.output_LD
output_L = args.output_L
output = args.output
num_workers = args.num_workers
lang = args.lang
model = args.model

analysis_data(input_file, output_LV, output_LD, output_L, output, num_workers, model, lang)
