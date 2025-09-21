import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import LlavaOnevisionForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from task2_runtime.surgvu_models import OrganClassifier, SurgToolClassifier
import task2_runtime.utils_llava_ov as utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="Video path")
parser.add_argument("--question", type=str, help="Question JSON path")
parser.add_argument("--gt_answers", type=str, help="Ground-truth answers JSON path")
parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--num_frames", type=int, default=5)

# Model paths
LLAVA_MODEL_ID = "./Models/llava-onevision-qwen2-7b-ov-hf"
ORGANS_MODEL_PATH = "./Models/organs_classifier/efficientnet_v2_s_v3/epoch=29_test_loss=0.04914_f1_avg=0.98064.ckpt"
TOOLS_MODEL_PATH = "./Models/tools_classifier/efficientnet_v2_s_smoothing_0.025/epoch=11_test_loss=0.12283_f1_avg=0.97795.ckpt"

# Target tools
target_tools = [
    "needle driver",
    "monopolar curved scissors",
    "force bipolar", 
    "clip applier",
    "cadiere forceps",
    "bipolar forceps",
    "vessel sealer",
    "permanent cautery hook/spatula",
    "prograsp forceps",
    "stapler",
    "grasping retractor",
    "tip-up fenestrated grasper",
]

# Target organs
target_organs = [
  'abdominal wall', 'bladder', 'gallbladder', 'pelvic lymph nodes',
  'rectum', 'sigmoid colon', 'unspecified', 'uterine horn',
]

target_tools = sorted(target_tools)
target_organs = sorted(target_organs)

def calculate_bleu_score(pred_answer, gt_answers):
    references_tokens = [_answer.split() for _answer in gt_answers]
    candidate_tokens = pred_answer.split()
    
    # BLEU Score
    smoothing_function = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    bleu = sentence_bleu(references_tokens, candidate_tokens, 
                         weights=weights, smoothing_function=smoothing_function)

    return bleu

if __name__ == "__main__":
    args = parser.parse_args()

    # load video
    video_clip = utils.read_video_decord(
        args.video_path, num_frames=args.num_frames, 
        crop_side=True, crop_bottom=True)

    # load question and ground-truth answers
    with open(args.question, 'r') as f:
        question_data = json.load(f)
        
    with open(args.gt_answers, 'r') as f:
        answer_data = json.load(f)

    # define quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # load the LLaVA model
    llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quantization_config, 
        device_map="cuda")

    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    # load tools and organs classifiers
    # tools classifier
    tools_model = SurgToolClassifier(model_name="efficientnet_v2_s")
    tools_model.load_from_ckpt(TOOLS_MODEL_PATH)

    tools_model = tools_model.to("cuda")
    tools_model = tools_model.eval()

    # organs classifier
    organs_model = OrganClassifier(model_name="efficientnet_v2_s")
    organs_model.load_from_ckpt(ORGANS_MODEL_PATH)

    organs_model = organs_model.to("cuda")
    organs_model = organs_model.eval()

    # predict tools and organs
    organ_name = utils.predict_organs(args.video_path, organs_model, target_organs)
    tools_all_frames = utils.predict_tools(
        args.video_path, tools_model, target_tools, num_frames=args.num_frames)
    
    tools_description = f"The organ being manipulated is the {organ_name}."
    for idx in range(args.num_frames):
        print(f"Frame {idx}: Detected tools: {tools_all_frames[idx]}")
        tools_description += f"Surgical tools in frame {idx}: {tools_all_frames[idx]}\n"
    
    # inference
    video_description = utils.get_video_description(
        model=llava_model, processor=llava_processor, 
        video_clip=video_clip, tools_present=tools_description, 
        max_new_tokens=args.max_new_tokens)

    video_description = tools_description + video_description
    
    print("Video description: ", video_description)
    
    final_answer = utils.get_answer(
        model=llava_model, processor=llava_processor, 
        video_description=video_description, video_clip=video_clip, 
        question_data=question_data)
    
    bleu_score = calculate_bleu_score(pred_answer=final_answer, gt_answers=answer_data)

    print(f"Question: {question_data}")
    print(f"Answer: {final_answer}")
    print(f"BLEU: {bleu_score}")