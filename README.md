# Dynamically Allocated Interval Based Generative Linguistic Steganography with Roulette Wheel

- Thanks to the editor and professional reviewers of *"Applied Soft computing"* for providing valuable comments on our paper!

<img src=https://github.com/WangYH-BUPT/DAIRstega/blob/master/figs/1.jpg width=83% />

## 1. Directory of repository 

```
·
├── LLMs               #(Selection of base LLMs. This paper uses LLaMA2-7B and LLaMA2-13B models for experiments.)
│   └── README.md
│
├── Stegos             #(DAIRstega generates steganotext with different payloads.)
│   ├── a8_b10_bpw110.csv
│   ├── a8_b05_bpw189.csv
│   ├── a16_b10_bpw113.csv
│   ├── a16_b05_bpw244.csv
│   ├── a32_b10_bpw111.csv
│   ├── a32_b05_bpw256.csv
│   └── README.md
│
├── bert-base-uncased  #(The BERTscore evaluation index requires the model to be loaded.)
│   └── README.md
│
├── bit_stream         #(A pseudo-random bitstream containing 1 million 0 and 1 is used to simulate the secret.)
│   ├── bit_stream.txt
│   └── README.md
│
├── figs               #(Including images from the repository description)
│   ├── 1.jpg
│   └── 2.jpg
│
├── finetune_data      #(Data for fine-tuning LLMs, not necessary)
│   ├── data.zip
│   └── README.md
│
├── ft-model           #(Fine-tuned LLMs, not necessary)
│   ├── adapter_config.json
│   ├── adapter_config.bin
│   └── README.md
│
├── steganalysis       #(Steganalysis methods to evaluate the anti-steganalysis of DAIRstega)
│   ├── Example
│   │   ├── GE.py
│   │   ├── TextCNN.py
│   │   ├── TextGE.py
│   │   ├── TextRNN.py
│   │   ├── data.py
│   │   ├── data_prepare.py
│   │   └── run.py
│   └── README.md
│
├── templates          #(Settings of prompts.)
│   ├── alpaca.json
│   ├── alpaca_legacy.json
│   ├── alpaca_short.json
│   ├── vigogne.json
│   └── README.md
│
├── utils              #(Settings of prompts.)
│   ├── __init__.py
│   ├── callbacks.py
│   ├── prompter.py
│   └── README.md
│
├── DAIRstega_Torch.py               #(Generation codes of the DAIRstega scheme (pytorch))
├── README.md
├── export_hf_checkpoint.py          #(Merge original LLM and LoRA models, not necessary)   
├── export_state_dict_checkpoint.py  #(Merge original LLM and LoRA models, not necessary)  
├── finetune.py                      #(Codes for fine-tuning LLMs, not necessary)
├── metrics_discourse.py             #(Metrics related to semantic concealment)
├── metrics_distributed.py           #(Metrics related to perceptual concealment and statistical concealment)
└── requirements.txt                 #(Necessary environment for the project)
```

## 2. Conda Environment

- python 3.8
- mindspore
- pytorch
- transformers
- peft
- `pip install -r requirements.txt`



## 3. Model configuration

```
base_model: str = "./LLM/LLaMA2-7B"
load_8bit: bool = False
lora_used: bool = False

if lora_used:
  if device == "cuda:0":
    model = LlamaForCausalLM.from_pretrained(base_model,
                                             load_in_8bit=load_8bit,
                                             torch_dtype=torch.float16,
                                             device_map="auto")

    model = PeftModel.from_pretrained(model,
                                      lora_weights,
                                      torch_dtype=torch.float16)

tokenizer = LlamaTokenizer.from_pretrained(base_model)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
        model.half()

model.eval()
```

## 4. Fine-tuning LLMs

DAIRstega does not depend on whether the model is fine-tuned, ***so this part is not necessary***. 

- If the original LLM is connected, just set: `lora_used: bool = False`.

- If you want to access a fine-tuning model, you can first construct the fine-tuning data you want to generate for a specific field to get the `.json`:

```
[
  {
    "instruction": "Compose a thesis statement for an essay about how the current education system could be improved.",
    "input": "",
    "output": "While the current education system has made significant strides in providing education to all, there is still much that can be done to improve the quality and efficacy of teaching; these improvements include incorporating new teaching methods, embracing technology, providing teachers with better resources, and focusing on developing critical thinking and problem-solving skills."
  },
...
]
```

Then, use this `json` file to fine-tune the specified LLM:

```
python finetune.py
```

```
lora_used: bool = True
model = PeftModel.from_pretrained(model,
                                  lora_weights,
                                  torch_dtype=torch.float16)
```


## 5. Embedding of DAIRstega

```
def DAIRstega_embedding(
        instruction,
        input=None,
        temperature=0.7,
        top_p=0.75,
        top_k=100,
        num_beams=1,
        max_new_tokens=512,
        stream_output=False,
        count=count,
        PRECISION=48,
        map=map,  # sqrt / sqrt3 / linear
        **kwargs
):

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        # temperature=temperature,
        top_p=top_p,
        # top_k=top_k,
        # num_beams=num_beams,
        early_stopping=True,
        do_sample=False,
        **kwargs
    )

    # GenerationConfig
    generate_params = {"input_ids": input_ids,
                       "generation_config": generation_config,
                       "return_dict_in_generate": True,
                       "output_scores": True,
                       "max_new_tokens": max_new_tokens}

    # -------------- Without streaming --------------
    with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
        bit_stream = f.read().strip()
        bit_stream += bit_stream
    bit_index = int(torch.randint(0, high=1000, size=(1,)))  # or 0

    with torch.no_grad():
        start = time.time()
        stega_text, stega_bits = [], ''

        for i in range(max_new_tokens - 1):
            if '</s>' in stega_text:
                break

            generation_output1 = model(input_ids)
            log_prob = generation_output1.logits
            prob = torch.softmax(log_prob, dim=-1)[:, -1, :].reshape(-1)

            prob = prob / prob.sum()
            prob, indices = prob.sort(descending=True)

            bit_tmp = 0
            PRECISION = PRECISION
            max_val = 2 ** PRECISION  # num of intervals
            cur_interval = [0, max_val]  # bottom inclusive, top exclusive
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range

            if prob[-1] < cur_threshold:
                k = max(2, (prob < cur_threshold).nonzero()[0].item())
                prob = prob[:k]
                indices = indices[:k]

            prob = prob[:top_k]
            indices = indices[:top_k]

            if map == "sqrt":
                prob = torch.round(torch.sqrt(prob), decimals=4)
            elif map == "sqrt3":
                prob = torch.pow(prob, 1 / 3)
            elif map == "sqrt4":
                prob = torch.pow(prob, 1 / 4)
            prob = prob / prob.sum()
            prob = prob.double()
            prob *= cur_int_range
            prob = prob.round().long()

            cum_probs = prob.cumsum(0)
            overfill_index = (cum_probs > cur_int_range).nonzero()  # tensor([[299]])

            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
            cum_probs += cur_int_range - cum_probs[-1]

            cum_probs += cur_interval[0]

            message_bits = bit_stream[bit_index: bit_index + PRECISION]
            message_bits = [int(_) for _ in message_bits]
            message_idx = bits2int(reversed(message_bits))
            selection = (cum_probs > message_idx).nonzero()[0].item()

            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, PRECISION)))
            new_int_top_bits_inc = list(reversed(int2bits(new_int_top - 1, PRECISION)))

            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            prev = indices[selection].view(1, 1)

            gen = int(prev)
            input_ids = torch.cat([input_ids, torch.LongTensor([[gen]]).to(device)], dim=1).to(device)
            stega_bits += bit_stream[bit_index:bit_index + num_bits_encoded]
            bit_index += num_bits_encoded

            if gen == 29889:
                # print(f"{gen},{tokenizer.decode(gen)}")
                count -= 1
                if 0 == count:
                    break

    end = time.time()
    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    costTime = end-start

    print(stega_bits)
    return output, stega_bits, costTime
```

## 6. Stegos generated by DAIRstega

Finally, the steganographic text containing secret information can be generated, for example:

- ***Instruction***: The car with the camera is driving down an icy road. A dark colored car tries to merge to the left lane. The dark car lost control of direction. The dark car hits on the roadside. Could the accident be prevented if the roads are marked clearly?
- ***LLMs***: LLaMA2-7B
- ***Cover text***: It is possible that the accident could have been prevented if the roads were marked clearly. When driving on icy roads, it is important to have clear signage and markings to help guide drivers and prevent accidents. If the dark car had seen the clear markings on the road, it may have been able to avoid losing control and hitting the car with the camera. Additionally, clear signage can help drivers slow down and be more cautious when driving on icy roads, which can also help prevent accidents.
- ***Stego text***: It is possible that the accident could have been prevented if the roads were marked clearly. When driving on icy roads, it is important to have clear signage and markings to help guide drivers and prevent accidents. If the dark car had seen the clear markings on the road, it may have been able to avoid losing control and hitting the car with the camera. Additionally, clear signage and markings can help drivers slow down and be more cautious when driving on icy roads, which can also help prevent accidents.
- ***Secret length***: 11 bits
- ***BERTscore***: 99.48

## Thanks again to the editor and professional reviewers of *"Applied Soft computing"* for providing valuable comments on our paper!

<img src=https://github.com/WangYH-BUPT/DAIRstega/blob/master/figs/2.jpg width=20% />
