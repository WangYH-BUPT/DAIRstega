# Dynamically Allocated Interval Based Generative Linguistic Steganography with Roulette Wheel

- **Thanks to the editor and professional reviewers of *"Applied Soft Computing"* for providing valuable comments on our paper!**

<img src=https://github.com/WangYH-BUPT/DAIRstega/blob/master/figs/1.jpg width=83% />

## 1. Conda Environment

- python 3.8
- mindspore 2.2
- mindformers 1.1.0
- mindpet 1.0.4
- peft
- `pip install -r requirements.txt`

## 2. Directory of repository 

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
├── DAIRstega.py               #(Generation codes of the DAIRstega scheme (MindSpore))
├── DAIRstega_t.py               #(Generation codes of the DAIRstega scheme (torch))
├── README.md
├── export_hf_checkpoint.py          #(Merge original LLM and LoRA models, not necessary)   
├── export_state_dict_checkpoint.py  #(Merge original LLM and LoRA models, not necessary)  
├── finetune.py                      #(Codes for fine-tuning LLMs, not necessary)
├── metrics_discourse.py             #(Metrics related to semantic concealment)
├── metrics_distributed.py           #(Metrics related to perceptual concealment and statistical concealment)
└── requirements.txt                 #(Necessary environment for the project)
```
