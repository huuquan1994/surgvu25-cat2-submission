## Preparation
1. Download the LLaVA-Onevision 7B model from https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf to this directory.
    ```bash
    git lfs install
    git clone https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf

    # remove .git in llava-onevision-qwen2-7b-ov-hf to reduce size for docker image
    cd llava-onevision-qwen2-7b-ov-hf
    rm -rf .git
    ```

2. Dowload tool and organ classifiers ([GDrive link](https://drive.google.com/drive/folders/1Fbnf1htcuoRPk3iGnnMliPTs9ULSkdPT?usp=sharing)), and put them to this directory.

The `Models` directory should look like below:
```bash
Models
└── llava-onevision-qwen2-7b-ov-hf
└── tools_classifier
└── organs_classifier
```