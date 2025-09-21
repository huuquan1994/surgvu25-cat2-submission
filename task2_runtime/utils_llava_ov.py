from decord import VideoReader, cpu
import numpy as np
import cv2
import albumentations as A
import torch

def read_video_decord(video_path, num_frames=10, crop_side=True, side_margin=192, 
                      crop_bottom=False, is_blur=False, k_size=51, bottom_margin=45):
    """Decode the video with Decord decoder
    """
    if crop_bottom==True and is_blur==True:
        raise Exception(f"crop_bottom and is_blur can't both be True")
    
    vr = VideoReader(uri=video_path, ctx=cpu(0)) # you need to install from source to use gpu ctx
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    if crop_side:
        new_frames = []
        for frame in frames:
            new_frames.append(frame[:, side_margin:-side_margin, :])

        frames = np.asarray(new_frames)

    if crop_bottom:
        new_frames = []
        for frame in frames:
            new_frames.append(frame[:-bottom_margin, :, :])

        frames = np.asarray(new_frames)
        
    if is_blur:
        for idx, frame in enumerate(frames):
            blurred_img = cv2.GaussianBlur(frame, (k_size, k_size), 0)
            mask = np.zeros_like(frame)
            mask[:-bottom_margin, :, :] = 1
            frame = np.where(mask, frame, blurred_img)
            frames[idx] = frame
    
    return frames

def get_inputs(video_clip, processor, tools_present):
    system_promt = """
    You are an excellent surgeon. 
    Your task is to answer questions about endoscopic surgery performed by surgical robots. 
    """

    _conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_promt},
                ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""The surgical tools in the video are: 
                {tools_present}. 
                Give me a very long and detailed description of the video, including the organs that being manipulated, the actions that being performed, and the purpose of each surgical tool.
                """},
                {"type": "video"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(_conversation, add_generation_prompt=False)
    batch = processor(
        text=prompt,
        videos=video_clip,
        return_tensors="pt",
    )

    return batch

def get_video_description(model, processor, video_clip, tools_present, search_key="assistant\n",
                          max_new_tokens=2048):
    inputs = get_inputs(video_clip, processor=processor, tools_present=tools_present)
    processor.batch_decode(inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # inference
    inputs = inputs.to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    description = outputs[0]
    description = description[description.find(search_key) + len(search_key):]

    return description

def get_answer(model, processor, video_description, video_clip, question_data, 
               search_key="assistant\n", max_new_tokens=2048):
    system_promt = """
    You are a helpful assistant. Your answers must be concise and within one short sentence.
    
    The examples below showing how you should response to questions:
    For example, 'Is a AAA driver being used here?' and you found no AAA driver --> 'No AAA driver is being used.'
    For example, 'What type of BBB forceps is mentioned?' and you found a BBB forceps --> 'The type of forceps mentioned is BBB forceps.'
    For example, 'What organ is being manipulated?' and you found the organ is CCC --> 'The organ being manipulated is the CCC.'
    For example, 'Is a DDD applier among the listed tools?' and you found a DDD applier --> 'Yes, a DDD applier is listed.'
    And so on ...
    """

    _conversation = [
        {
            "role": "system",
            "content": [
            {"type": "text", "text": system_promt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                The examples below showing how you should response to questions:
                For example, 'Is a AAA driver being used here?' and you found no AAA driver --> 'No AAA driver is being used.'
                For example, 'What type of BBB forceps is mentioned?' and you found a BBB forceps --> 'The type of forceps mentioned is BBB forceps.'
                For example, 'What organ is being manipulated?' and you found the organ is CCC --> 'The organ being manipulated is the CCC.'
                For example, 'Is a DDD applier among the listed tools?' and you found a DDD applier --> 'Yes, a DDD applier is listed.'
                And so on ...
                
                Given the description:
                The video is describing an endoscopic surgery or a laparoscopic surgery. {video_description}.
                
                Answer in one short sentence.
                Now, answer the question: {question_data}
                """},
                {"type": "video"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(_conversation, add_generation_prompt=False)
    inputs = processor(text=prompt, videos=video_clip, return_tensors="pt")
    
    inputs = inputs.to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    answer = outputs[0]
    answer = answer[answer.find(search_key) + len(search_key):]

    return answer

def normalize_data(tensor_images: torch.Tensor, 
                    mean=[0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]):
    # tensor_images: (Batch, H, W, C), dtype=uint8
    assert tensor_images.ndim == 4, "input tensors should be 4D (Batch, H, W, C)"
    assert tensor_images.shape[-1] == 3, "input tensors should have 3 channels"

    tensor_images = tensor_images.permute(0, 3, 1, 2)  # Batch, C, H, W
    tensor_images = tensor_images.float() / 255.0

    mean = torch.tensor(mean, device=tensor_images.device).view(1, 3, 1, 1) # per-channel normalize
    std  = torch.tensor(std, device=tensor_images.device).view(1, 3, 1, 1) # per-channel normalize

    tensor_images = (tensor_images - mean) / std
    
    return tensor_images

def predict_tools(video_path, model, target_tools, thr_score=0.5, img_size=512, num_frames=10, device="cuda"):
    # define transform
    _transforms = [
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
    ]
    _aug_func = A.Compose(_transforms)

    # load video
    blurred_frames = read_video_decord(str(video_path), num_frames=num_frames, is_blur=True)

    # resize
    images = [_aug_func(image=image)['image'] for image in blurred_frames]
    images = np.asarray(images)
    images = torch.from_numpy(images)
    
    images = normalize_data(images)
    images = images.to(device)

    with torch.inference_mode():
        outputs = model.forward(images)

    outputs = outputs.detach().cpu().numpy()
    preds = outputs > thr_score

    pred_tools = []
    for pred in preds:
        pred_tools.append(', '.join(np.array(target_tools)[pred].tolist()))

    return pred_tools

def predict_organs(video_path, model, target_organs, img_size=512, num_frames=30, device="cuda"):
    # define transform
    _transforms = [
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
    ]
    _aug_func = A.Compose(_transforms)

    # load video
    blurred_frames = read_video_decord(str(video_path), num_frames=num_frames, is_blur=True)

    # resize
    images = [_aug_func(image=image)['image'] for image in blurred_frames]
    images = np.asarray(images)
    images = torch.from_numpy(images)
    
    images = normalize_data(images)
    images = images.to(device)

    with torch.inference_mode():
        outputs = model.forward(images)

    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

    values, counts = np.unique(preds, return_counts=True)
    pred_id = values[np.argmax(counts)]
    
    return target_organs[pred_id]