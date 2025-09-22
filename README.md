## 1st Place Solution - Category 2: Surgical Visual Question Answering - SurgVU Challenge (MICCAI 2025)
**Effective Surgical Visual Question Answering Without Ground-Truth Video Descriptions**

**Team**: Capybara <br>
**Member**: Quan Huu Cap <br>
**Affiliation:**: [Aillis, Inc.](https://aillis.jp/), Tokyo, Japan <br>
**Solution Report**: [Link](solution_report/Team_Capybara_report.pdf)

### Required env
- Ubuntu with Docker
- Python 3.10
- CUDA 11.8
  
### Installation
```bash
pip install --no-cache-dir -r requirements.txt
```

### Video pre-processing
For each video frame, the black margins on the left and right were cropped, and the tool list area at the bottom was either cropped or obscured using Gaussian blur. See [utils_llava_ov.py](task2_runtime/utils_llava_ov.py) for details.

### Models download
See [Models prepration](Models/README.md) for more details.

### Inference
The inference example is as follows:
```bash
python inference_sample.py \
  --video_path input-sample/case127.mp4 \
  --question input-sample/case127_question.json \
  --gt_answers input-sample/case127.json
```
The outputs should be:
```
Question: What organ is being manipulated?
Answer: The organ being manipulated is the uterine horn.
BLEU: 1.0
```

### Corrected version: Category 2 sample set (11 videos)
Some answers in the [public sample set (11 videos)](https://surgvu25.grand-challenge.org/data-description/) are wrong and need correction. I corrected them myself to validate the method. Feel free to download them at ([GDrive - Corrected Videos](https://drive.google.com/file/d/17sOEzW8FI9VJxY0yapWDtCjXM-91aNI2/view?usp=sharing))

### Build docker image
Double check the [Dockerfile](Dockerfile) before running this command.
```bash
docker build --platform=linux/amd64 -t surgvu25-cat2-submit:ov-tools-organs-5fr .
```

### Save the docker image for submission
```bash
docker save surgvu25-cat2-submit:ov-tools-organs-5fr | gzip > surgvu25-cat2-submit-ov-tools-organs-5fr.tar.gz
```

### Citation
```
@article{quancap25surgvucat2,
  title   = {Effective Surgical Visual Question Answering Without Ground-Truth Video Descriptions},
  author  = {Quan Huu Cap},
  journal = {arXiv preprint},
  year    = {2025}
}
```