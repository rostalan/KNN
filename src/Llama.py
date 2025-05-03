import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType


class Llama_model:
    def __init__(self):
        self.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.output = "checkpoints"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, dataset_loc:str, valid_data_location:str):

        train_dataset, test_dataset = self.parse_dataset(dataset_loc)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )


        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_config=bnb_config,
            device_map="auto"
        )


        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=self.output,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=500,
            fp16=True,
            save_total_limit=6,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        trainer.train(resume_from_checkpoint="checkpoints/checkpoint-9000")
        trainer.save_model(self.output)
        self.tokenizer.save_pretrained(self.output)

    def _preprocess(self,example):
        prompt = f"### User:\nShrň následující text:\n\n{example['text']}\n\n### Assistant:\n"
        full_text = prompt + example["abstract"]
        return {"text": full_text}
    
    def _tokenize(self,example):
        tokens = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=1024,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def parse_dataset(self, path_to_dataset:str):

        dataset = load_dataset("json", data_files=path_to_dataset, split="train")
        dataset = dataset.select(range(20000))
        dataset = dataset.train_test_split(test_size=0.1)

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        train_dataset = train_dataset.map(self._preprocess)
        test_dataset = test_dataset.map(self._preprocess)

        

        train_dataset = train_dataset.map(self._tokenize, batched=True)
        test_dataset = test_dataset.map(self._tokenize, batched=True)

        # Odstranění nepotřebných sloupců ???
        #train_dataset = train_dataset.remove_columns(["text", "abstract"])
        #test_dataset = test_dataset.remove_columns(["text", "abstract"])

        return train_dataset, test_dataset



    def summarize(self, data_location: str):
        tokenizer = AutoTokenizer.from_pretrained(data_location)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto")
        model = PeftModel.from_pretrained(base_model, data_location)
        model.eval()
        text = "Řada zaměstnanců se pro získání několika volných dní v celku snaží využít i státní svátky, které kombinuje s krátkou několikadenní dovolenou. Například začátek července, kde se sejdou dva státní svátky za sebou, dává k takovému řešení velkou příležitost. Červencové \"supervolno\" vylidní až polovinu firem, úřadů a institucí v Čechách i na Moravě, potvrzuje exkluzivní reprezentativní průzkum společnosti Wincott People.\nBezmála 50 procent zaměstnanců totiž Den slovanských věrozvěstů Cyrila a Metoděje (5. 7.) a Den upálení mistra Jana Husa (6. 7.) spojuje s oblibou s dalším jedním či více dny řádné dovolené.\nČeši si plánují na tyto červencové svátky především rodinné výlety (30 procent dotázaných), spojí je s řádnou letní dovolenou (16 procent), případně volno využijí pro vylepšení bydlení, pro práci doma, na chatě, chalupě nebo na zahrádce (20 procent).Letní odstávky provozu spojené se závodní dovolenou jsou typické hlavně pro automobilový průmysl.Adrian Suchánek, agentura Wincott People Se třinácti civilními a náboženskými svátky se Česká republika v EU řadí k průměru. Více svátků mají například Německo a Slovensko (16 volných dní), naopak nejméně slaví Nizozemci (9 svátků). Za námi jsou překvapivě i silně katoličtí Poláci s pouhými deseti převážně církevními svátky.\nMinisterstvo financí spočítalo, že jeden státní svátek ekonomiku připraví až o 0,4 procenta ročního hrubého domácího produktu. Červencové \"supervolno\" by tak odhadem mohlo českou ekonomiku při polovičním výkonu firem přijít až na 18,5 miliardy korun. Ekonomové s tím ale tak docela nesouhlasí.\nSice se méně pracuje, ale zpravidla se na druhé straně více spotřebovává – lidé cestují, utrácejí více za služby ubytování nebo pohostinství, nakupují. Řada majitelů chat a rekreačních objektů využije volné dny k přestavbě či vylepšení svých nemovitostí, což je spojeno i s investicemi do stavebního materiálu.\nPři plánování dovolené se musí přes 49 procent zaměstnanců přizpůsobit potřebám svých zaměstnavatelů.\nČást dovolené tak mohou Češi čerpat buď po dohodě, nebo přímo v termínu určeném zaměstnavatelem. Ve větší míře se to týká mužů než žen a spíše obyvatel větších měst. Průzkum Wincott People také dokazuje, že téměř 25 procent pracovníků má problém si v práci vzít souvislé dva týdny dovolené, ačkoli toto právo zakotvuje zákoník práce.Kde nejsou potřeba kvalifikované síly, je možné pracovníky na dovolené nahradit brigádníky, což některé podniky dělají právě v létě.Adrian Suchánek Více než pětině pracovníků určuje zaměstnavatel takzvané celozávodní volno. V 15 procentech případů v pravidelných termínech.\n\"Letní odstávky provozu spojené se závodní dovolenou jsou typické hlavně pro automobilový průmysl. Ty letošní se překryjí s červencovými svátky a připadnou na 4. až 17. července,\" upozornil Adrian Suchánek, personální ředitel agentury Wincott People.\n\"U strojírenských podniků pak záleží, zda má klient zakázky, na kterých se musí pracovat. V tom případě se se zaměstnanci snažíme buď dohodnout na dovolené v jiném termínu, nebo zakázku částečně pokrýváme přesčasy, kdy kupříkladu dva lidé mají dovolenou, další přesčasy a pak se to vymění. Kde nejsou potřeba kvalifikované síly, je možné pracovníky na dovolené nahradit brigádníky, což některé podniky dělají právě v létě,\" dodal Suchánek."
        print(self.generate_summary(model,tokenizer,text))

    def summarize_raw(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        tokenizer.pad_token = tokenizer.eos_token


        model = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto")
        model.eval()
        text = "Řada zaměstnanců se pro získání několika volných dní v celku snaží využít i státní svátky, které kombinuje s krátkou několikadenní dovolenou. Například začátek července, kde se sejdou dva státní svátky za sebou, dává k takovému řešení velkou příležitost. Červencové \"supervolno\" vylidní až polovinu firem, úřadů a institucí v Čechách i na Moravě, potvrzuje exkluzivní reprezentativní průzkum společnosti Wincott People.\nBezmála 50 procent zaměstnanců totiž Den slovanských věrozvěstů Cyrila a Metoděje (5. 7.) a Den upálení mistra Jana Husa (6. 7.) spojuje s oblibou s dalším jedním či více dny řádné dovolené.\nČeši si plánují na tyto červencové svátky především rodinné výlety (30 procent dotázaných), spojí je s řádnou letní dovolenou (16 procent), případně volno využijí pro vylepšení bydlení, pro práci doma, na chatě, chalupě nebo na zahrádce (20 procent).Letní odstávky provozu spojené se závodní dovolenou jsou typické hlavně pro automobilový průmysl.Adrian Suchánek, agentura Wincott People Se třinácti civilními a náboženskými svátky se Česká republika v EU řadí k průměru. Více svátků mají například Německo a Slovensko (16 volných dní), naopak nejméně slaví Nizozemci (9 svátků). Za námi jsou překvapivě i silně katoličtí Poláci s pouhými deseti převážně církevními svátky.\nMinisterstvo financí spočítalo, že jeden státní svátek ekonomiku připraví až o 0,4 procenta ročního hrubého domácího produktu. Červencové \"supervolno\" by tak odhadem mohlo českou ekonomiku při polovičním výkonu firem přijít až na 18,5 miliardy korun. Ekonomové s tím ale tak docela nesouhlasí.\nSice se méně pracuje, ale zpravidla se na druhé straně více spotřebovává – lidé cestují, utrácejí více za služby ubytování nebo pohostinství, nakupují. Řada majitelů chat a rekreačních objektů využije volné dny k přestavbě či vylepšení svých nemovitostí, což je spojeno i s investicemi do stavebního materiálu.\nPři plánování dovolené se musí přes 49 procent zaměstnanců přizpůsobit potřebám svých zaměstnavatelů.\nČást dovolené tak mohou Češi čerpat buď po dohodě, nebo přímo v termínu určeném zaměstnavatelem. Ve větší míře se to týká mužů než žen a spíše obyvatel větších měst. Průzkum Wincott People také dokazuje, že téměř 25 procent pracovníků má problém si v práci vzít souvislé dva týdny dovolené, ačkoli toto právo zakotvuje zákoník práce.Kde nejsou potřeba kvalifikované síly, je možné pracovníky na dovolené nahradit brigádníky, což některé podniky dělají právě v létě.Adrian Suchánek Více než pětině pracovníků určuje zaměstnavatel takzvané celozávodní volno. V 15 procentech případů v pravidelných termínech.\n\"Letní odstávky provozu spojené se závodní dovolenou jsou typické hlavně pro automobilový průmysl. Ty letošní se překryjí s červencovými svátky a připadnou na 4. až 17. července,\" upozornil Adrian Suchánek, personální ředitel agentury Wincott People.\n\"U strojírenských podniků pak záleží, zda má klient zakázky, na kterých se musí pracovat. V tom případě se se zaměstnanci snažíme buď dohodnout na dovolené v jiném termínu, nebo zakázku částečně pokrýváme přesčasy, kdy kupříkladu dva lidé mají dovolenou, další přesčasy a pak se to vymění. Kde nejsou potřeba kvalifikované síly, je možné pracovníky na dovolené nahradit brigádníky, což některé podniky dělají právě v létě,\" dodal Suchánek."
        print(self.generate_summary(model,tokenizer,text))

    def generate_summary(self,model, tokenizer, text: str, max_new_tokens: int = 200):
        prompt = f"### User:\nShrň následující text:\n\n{text}\n\n### Assistant:\n"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("### Assistant:")[-1].strip()