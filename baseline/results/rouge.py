import evaluate

rouge = evaluate.load('rouge')

with open("sumeczech_step50000.candidate", "r", encoding="utf-8") as pred_file:
    predictions = [line.strip() for line in pred_file.readlines()]
with open("sumeczech_step50000.gold", "r", encoding="utf-8") as ref_file:
    references = [line.strip() for line in ref_file.readlines()]

#assert len(predictions) == len(references), "ERROR"

#for i in range(10):
#    result = rouge.compute(predictions=[predictions[i]], references=[references[i]])
#    print(f"{i + 1}: {result}")

all_results = rouge.compute(predictions=predictions, references=references)

for x, y in all_results.items():
    print(f"{x}: {y:.4f}")