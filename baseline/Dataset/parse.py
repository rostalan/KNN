#Parse data from dataset.jsonl into tokens

import json
import stanza
from transformers import AutoTokenizer

stanza.download("cs")
nlp = stanza.Pipeline(lang="cs", processors="tokenize", use_gpu=False)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
input = "dataset.jsonl"   
output = "sumeczech.train"  

number = 0
data_list = []
file_index = 0

with open(input, "r", encoding="utf-8") as f:
    for line in f:
        #if number < 90000:
        #    number += 1
        #    continue
        data = json.loads(line)
        
        doc = nlp(data["text"])
        article_sentences = [sentence.text for sentence in doc.sentences]
        doc_summary = nlp(data["abstract"])
        summary_sentences = [sentence.text for sentence in doc_summary.sentences]
        
        # Tokenize
        tokenized_article = [tokenizer.tokenize(sent) for sent in article_sentences]
        tokenized_summary = [tokenizer.tokenize(sent) for sent in summary_sentences]
        data_list.append({
            "src": tokenized_article,  
            "tgt": tokenized_summary  
        })
        number += 1

        if number % 1000 == 0: #Log
            print(number)
        
        # Save every 10k
        if number % 10000 == 0:
            output_json = f"{output}.{file_index}.json"
            with open(output_json, "w", encoding="utf-8") as out_f:
                json.dump(data_list, out_f, ensure_ascii=False, indent=2)
            print(f"Saved to {output_json}")
            data_list = []
            file_index += 1

# IF not empty
if data_list:
    output_json = f"{output}.{file_index}.json"
    with open(output_json, "w", encoding="utf-8") as out_f:
        json.dump(data_list, out_f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_json}")

print(f"FIN")