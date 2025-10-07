import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

import os
import json
import argparse
import sys
from typing import List, Dict, Any
from PIL import Image
import PIL
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, Dataset

from vllm import LLM, SamplingParams
from transformers import AutoProcessor


# Aspect ratio mappings for image preprocessing
ASPECT_RATIO_1024 = {
    "0.25": [512.0, 2048.0], "0.26": [512.0, 1984.0], "0.27": [512.0, 1920.0], "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0], "0.33": [576.0, 1728.0], "0.35": [576.0, 1664.0], "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0], "0.48": [704.0, 1472.0], "0.5": [704.0, 1408.0], "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0], "0.6": [768.0, 1280.0], "0.68": [832.0, 1216.0], "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0], "0.82": [896.0, 1088.0], "0.88": [960.0, 1088.0], "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0], "1.07": [1024.0, 960.0], "1.13": [1088.0, 960.0], "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0], "1.38": [1152.0, 832.0], "1.46": [1216.0, 832.0], "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0], "2.0": [1408.0, 704.0], "2.09": [1472.0, 704.0], "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0], "2.89": [1664.0, 576.0], "3.0": [1728.0, 576.0], "3.11": [1792.0, 576.0],
    "3.62": [1856.0, 512.0], "3.75": [1920.0, 512.0], "3.88": [1984.0, 512.0], "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512 = {
    "0.25": [256.0, 1024.0], "0.26": [256.0, 992.0], "0.27": [256.0, 960.0], "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0], "0.33": [288.0, 864.0], "0.35": [288.0, 832.0], "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0], "0.48": [352.0, 736.0], "0.5": [352.0, 704.0], "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0], "0.6": [384.0, 640.0], "0.68": [416.0, 608.0], "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0], "0.82": [448.0, 544.0], "0.88": [480.0, 544.0], "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0], "1.07": [512.0, 480.0], "1.13": [544.0, 480.0], "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0], "1.38": [576.0, 416.0], "1.46": [608.0, 416.0], "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0], "2.0": [704.0, 352.0], "2.09": [736.0, 352.0], "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0], "2.89": [832.0, 288.0], "3.0": [864.0, 288.0], "3.11": [896.0, 288.0],
    "3.62": [928.0, 256.0], "3.75": [960.0, 256.0], "3.88": [992.0, 256.0], "4.0": [1024.0, 256.0],
}


def var_center_crop(pil_image, crop_size_dict):
    """Resize and center crop image to match aspect ratio."""
    aspect_ratio = pil_image.size[0] / pil_image.size[1]
    closet_ratio = min(crop_size_dict.keys(), key=lambda x: abs(float(x) - aspect_ratio))
    crop_w, crop_h = map(int, crop_size_dict[closet_ratio])

    scale = max(crop_w / pil_image.size[0], crop_h / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=PIL.Image.LANCZOS)

    crop_left = (pil_image.size[0] - crop_w) // 2
    crop_upper = (pil_image.size[1] - crop_h) // 2
    crop_right = crop_left + crop_w
    crop_lower = crop_upper + crop_h
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def init_vllm(model_path: str, tp: int, dtype: str, gpu_util: float, max_len: int):
    """Initialize vLLM engine."""
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        dtype=dtype,
        gpu_memory_utilization=gpu_util,
        trust_remote_code=True,
        max_model_len=max_len,
    )
    return llm


def vllm_generate_batch(llm: LLM,
                        processor: AutoProcessor,
                        batch_msgs: List[List[Dict[str, Any]]],
                        batch_images: List[List[Image.Image]],
                        max_new_tokens: int,
                        temperature: float = 0.0) -> List[str]:
    """Generate responses using vLLM in batch mode."""
    prompts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in batch_msgs
    ]
    
    sp = SamplingParams(
        temperature=temperature,
        top_p=1,
        max_tokens=max_new_tokens,
        stop=["��", "\n��", "📐\n"]
    )
    
    inputs = []
    for text, imgs in zip(prompts, batch_images):
        inputs.append({"prompt": text, "multi_modal_data": {"image": imgs}})
    
    outs = llm.generate(inputs, sp)
    return [o.outputs[0].text if o.outputs else "" for o in outs]


def create_evaluation_tasks(dataset, prefix, img_size):
    """
    Create evaluation tasks for each item, model, and QA pair in the dataset.
    Returns tasks with preprocessed images for vLLM efficiency.
    """
    tasks = []
    
    for item_index, item in enumerate(dataset):
        # Find all model names by detecting keys with the specified prefix
        model_names = [k.replace(prefix, "") for k in item.keys() if k.startswith(prefix)]

        if not model_names:
            print(f"Warning: No model images found in item {item_index}, skipping")
            continue

        qa_list = item.get("qa_list", [])
        if not qa_list:
            print(f"Warning: No qa_list in item {item_index}, skipping")
            continue

        # Preprocess all model images
        model_images = {}
        for model_name in model_names:
            model_image = item.get(f"{prefix}{model_name}")
            if not isinstance(model_image, Image.Image):
                print(f"Warning: Invalid image for model {model_name} in item {item_index}, skipping")
                continue
            
            # Preprocess image
            img = model_image.convert("RGB")
            if img_size == 512:
                img = var_center_crop(img, ASPECT_RATIO_512)
            elif img_size == 1024:
                img = var_center_crop(img, ASPECT_RATIO_1024)
            
            model_images[model_name] = img

        # Create tasks for each model and QA pair
        for model_name in model_images.keys():
            for qa_index, qa_pair in enumerate(qa_list):
                question = qa_pair.get("question")
                ground_truth_answer = qa_pair.get("ground_truth_answer", qa_pair.get("answer"))

                if question is None or ground_truth_answer is None:
                    print(f"Warning: Malformed qa_pair in item {item_index}, skipping")
                    continue

                tasks.append({
                    "item_index": item_index,
                    "model_name": model_name,
                    "qa_index": qa_index,
                    "question": question,
                    "ground_truth_answer": ground_truth_answer,
                    "preprocessed_image": model_images[model_name],
                })

    return tasks


def evaluate_with_vllm(tasks, llm, processor, max_new_tokens):
    """
    Evaluate all tasks using vLLM in batch mode.
    Returns list of results with answers and corrections.
    """
    if not tasks:
        return []
    
    # Step 1: Batch answer all questions
    QA_PROMPT = "You are a vision QA assistant. Look at the image carefully and answer the question in the simplest words. Your answer must be the single, definitive, and deterministic response based exclusively on the visual information in the image. Do not infer or add outside information. If the question is not about the image, or the mentioned elements are not visible, return \"N/A\". Directly output the concise answer with no explanation.\nQuestion: "
    
    qa_msgs_batch = []
    qa_imgs_batch = []
    for task in tasks:
        img = task["preprocessed_image"]
        question = task["question"]
        qa_msgs_batch.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": QA_PROMPT + question}
            ]
        }])
        qa_imgs_batch.append([img])
    
    print(f"\nAnswering {len(tasks)} questions...")
    answers = vllm_generate_batch(llm, processor, qa_msgs_batch, qa_imgs_batch, max_new_tokens)
    
    # Step 2: Batch evaluate all answers
    EVAL_PROMPT = (
        "# Task: Evaluate if the model's response is correct.\n"
        "Based on the \"Question\", \"Ground Truth Answer\" and \"Model Response\" provided below, please determine if the \"Model Response\" is acceptable.\n\n"
        "# Evaluation Criteria:\n"
        "1. **Core Meaning Priority**: The model's response should capture the essential entity, action, or relation described in the Ground Truth Answer.\n"
        "   - Accept answers that correctly identify the main object(s), relation(s), or action(s), even if they omit secondary descriptors such as size, color, thickness, or position.\n"
        "   - Accept concise answers if they preserve the central meaning of the Ground Truth Answer.\n"
        "   - Accept alternative wording or paraphrases that do not contradict the core content.\n"
        "   - Reject answers that describe a different object, relation, or action than the ground truth.\n"
        "   - Reject answers that are \"N/A\".\n"
        "\n"
        "2. **Numerical Values**: A tolerance of ±10% is allowed for any numerical values compared to the \"Ground Truth Answer\".\n"
        "\n"
        "3. **Over- or under-specification**:\n"
        "   - Accept answers that are less detailed but still factually consistent with the ground truth.\n"
        "   - Accept answers that provide extra detail if this detail does not contradict the ground truth.\n"
        "   - Reject answers that omit or alter the critical subject/object or core relation.\n\n"
        "# Inputs:\n"
        "[Question]: {question}\n"
            "[Ground Truth Answer]: {ground_truth_answer}\n"
            "[Model Response]: {model_response}\n\n"
        "# Output Requirements:\n"
        "Please judge strictly according to the rules above and output only the single word \"Correct\" or \"Incorrect\". Do not include any other explanations, reasons, or punctuation."
    )

    eval_msgs_batch = []
    eval_imgs_batch = []
    for i, task in enumerate(tasks):
        eval_text = EVAL_PROMPT.format(
            question=task["question"],
            ground_truth_answer=task["ground_truth_answer"],
            model_response=answers[i]
        )
        eval_msgs_batch.append([{
            "role": "user",
            "content": [{"type": "text", "text": eval_text}]
        }])
        eval_imgs_batch.append([])

    print(f"Evaluating {len(tasks)} answers...")
    evaluations = vllm_generate_batch(llm, processor, eval_msgs_batch, eval_imgs_batch, max_new_tokens=16)
    
    # Compile results
    results = []
    for i, task in enumerate(tasks):
        is_correct = "Correct" in evaluations[i]
        results.append({
            "item_index": task["item_index"],
            "model_name": task["model_name"],
            "qa_index": task["qa_index"],
            "answer": answers[i],
            "Correction": is_correct,
            "question": task["question"],
            "ground_truth_answer": task["ground_truth_answer"],
        })
    
    return results


def process_evaluation_results(results, original_dataset):
    """
    Reorganize evaluation results back into the original dataset format.
    """
    results_by_item = defaultdict(lambda: defaultdict(dict))
    
    # Organize results by item_index and model_name
    for result in results:
        item_index = result["item_index"]
        model_name = result["model_name"]
        qa_index = result["qa_index"]
        
        results_by_item[item_index][model_name][qa_index] = {
            "answer": result["answer"],
            "Correction": result["Correction"],
            "question": result["question"],
            "ground_truth_answer": result["ground_truth_answer"],
        }

    # Build final results with per-model, per-sample weighted accuracy
    final_results = []
    for item_index, item in enumerate(original_dataset):
        new_item = dict(item)
        
        # Remove old result columns
        for key in ["response_list", "error", "new_column"]:
            if key in new_item:
                del new_item[key]
        
        # Get original qa_list to extract labels
        original_qa_list = item.get("qa_list", [])
        
        # Add new result columns and compute per-model, per-sample weighted accuracy
        if item_index in results_by_item:
            for model_name, qa_results_dict in results_by_item[item_index].items():
                # Convert dict to ordered list and preserve labels
                max_qa_index = max(qa_results_dict.keys()) if qa_results_dict else -1
                ordered_results = []
                
                # Track stats by label
                editing_correct = 0
                editing_total = 0
                maintain_correct = 0
                maintain_total = 0
                
                for i in range(max_qa_index + 1):
                    if i in qa_results_dict:
                        result_item = qa_results_dict[i].copy()
                        
                        # Add label from original qa_list
                        if i < len(original_qa_list):
                            label = original_qa_list[i].get("label", "editing")
                            result_item["label"] = label
                            
                            # Track by label
                            if label == "editing":
                                editing_total += 1
                                if result_item["Correction"]:
                                    editing_correct += 1
                            elif label == "maintain":
                                maintain_total += 1
                                if result_item["Correction"]:
                                    maintain_correct += 1
                        
                        ordered_results.append(result_item)
                    else:
                        ordered_results.append({
                            "answer": "Missing result",
                            "Correction": False,
                            "question": "",
                            "ground_truth_answer": "",
                            "label": "editing",
                            "error": "Result not found"
                        })
                        editing_total += 1
                
                new_item[f"{model_name}_qwen_list"] = ordered_results
                
                # Compute weighted accuracy: 0.1 * maintain_acc + 0.9 * editing_acc
                maintain_acc = (maintain_correct / maintain_total * 100) if maintain_total > 0 else 0.0
                editing_acc = (editing_correct / editing_total * 100) if editing_total > 0 else 0.0
                if maintain_total <= 0:
                    weighted_acc = editing_acc
                else:
                    weighted_acc = 0.1 * maintain_acc + 0.9 * editing_acc
                
                new_item[f"{model_name}_qwen_accuracy"] = weighted_acc
                new_item[f"{model_name}_qwen_editing_accuracy"] = editing_acc
                new_item[f"{model_name}_qwen_maintain_accuracy"] = maintain_acc
        
        final_results.append(new_item)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenVL Vision QA Evaluation Pipeline with vLLM")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Hugging Face dataset path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and processed dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to QwenVL model",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=5120,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=1024,
        choices=[512, 1024],
        help="Image resize size for preprocessing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process 20 samples",
    )
    parser.add_argument(
        "--output_repo_name",
        type=str,
        default=None,
        help="Hugging Face Hub repo name for upload (optional)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="output_image_",
        help="Prefix for model image columns",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_dataset(args.dataset_path, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.debug:
        print("DEBUG MODE: Processing only 20 samples")
        dataset = dataset.select(range(20))

    print(f"Total samples: {len(dataset)}")

    # Initialize vLLM and processor
    print(f"\nInitializing vLLM with {args.tensor_parallel_size} GPUs...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = init_vllm(args.model_path, args.tensor_parallel_size, args.dtype, args.gpu_mem_util, args.max_model_len)

    # Create evaluation tasks
    print("\nCreating evaluation tasks...")
    evaluation_tasks = create_evaluation_tasks(dataset, args.prefix, args.img_size)
    print(f"Total tasks created: {len(evaluation_tasks)}")

    if not evaluation_tasks:
        print("No valid evaluation tasks, exiting")
        sys.exit(1)

    # Run evaluation
    print("\nStarting evaluation with vLLM...")
    evaluation_results = evaluate_with_vllm(evaluation_tasks, llm, processor, args.max_new_tokens)

    # Reorganize results
    print("\nProcessing evaluation results...")
    final_results = process_evaluation_results(evaluation_results, dataset)

    # Create final dataset
    final_dataset = Dataset.from_list(final_results)

    # Save to local disk
    local_save_path = os.path.join(args.output_dir, "processed_dataset")
    print(f"\nSaving processed dataset to: {local_save_path}")
    final_dataset.save_to_disk(local_save_path)
    print("Dataset saved successfully")

    # Upload to Hugging Face Hub if specified
    if args.output_repo_name:
        print(f"\nUploading dataset to Hugging Face Hub: {args.output_repo_name}")
        print("Ensure you are logged in via 'huggingface-cli login'")
        try:
            final_dataset.push_to_hub(args.output_repo_name)
            print("Dataset uploaded successfully!")
        except Exception as e:
            print(f"Upload failed: {e}")

    print(f"\n✓ Evaluation complete!")
    print(f"Total samples processed: {len(final_results)}")
    print(f"Total QA evaluations: {len(evaluation_results)}")

    # Identify all models from the results
    all_models = set()
    for item in final_results:
        for key in item.keys():
            if key.endswith("_qwen_accuracy"):
                model_name = key[:-len("_qwen_accuracy")]
                all_models.add(model_name)
    
    all_models = sorted(all_models)
    
    if not all_models:
        print("Warning: No models found in results")
        sys.exit(0)

    print(f"\nFound {len(all_models)} model(s): {', '.join(all_models)}")

    # Compute accuracies for each model separately
    dataset_name = os.path.basename(args.dataset_path.strip("/"))
    
    for model_name in all_models:
        accuracy_key = f"{model_name}_qwen_accuracy"
        editing_key = f"{model_name}_qwen_editing_accuracy"
        maintain_key = f"{model_name}_qwen_maintain_accuracy"
        
        # 1. Global weighted accuracy: average of all sample weighted accuracies for this model
        sample_accuracies = [
            item[accuracy_key] for item in final_results 
            if accuracy_key in item
        ]
        global_accuracy = sum(sample_accuracies) / len(sample_accuracies) if sample_accuracies else 0.0
        
        # Also compute global editing and maintain accuracies
        editing_accuracies = [
            item[editing_key] for item in final_results 
            if editing_key in item
        ]
        maintain_accuracies = [
            item[maintain_key] for item in final_results 
            if maintain_key in item
        ]
        global_editing_acc = sum(editing_accuracies) / len(editing_accuracies) if editing_accuracies else 0.0
        global_maintain_acc = sum(maintain_accuracies) / len(maintain_accuracies) if maintain_accuracies else 0.0

        # 2. Group accuracy: average sample accuracies grouped by category for this model
        category_samples = defaultdict(lambda: {"weighted": [], "editing": [], "maintain": []})
        for item in final_results:
            if accuracy_key in item:
                category = item.get("category", "unknown")
                category_samples[category]["weighted"].append(item[accuracy_key])
                if editing_key in item:
                    category_samples[category]["editing"].append(item[editing_key])
                if maintain_key in item:
                    category_samples[category]["maintain"].append(item[maintain_key])
        
        group_accuracies = {}
        for category, accs in category_samples.items():
            group_accuracies[category] = {
                "accuracy": sum(accs["weighted"]) / len(accs["weighted"]) if accs["weighted"] else 0.0,
                "editing_accuracy": sum(accs["editing"]) / len(accs["editing"]) if accs["editing"] else 0.0,
                "maintain_accuracy": sum(accs["maintain"]) / len(accs["maintain"]) if accs["maintain"] else 0.0,
                "num_samples": len(accs["weighted"])
            }

        # Print results for this model
        print("\n" + "="*50)
        print(f"ACCURACY REPORT - {model_name.upper()}")
        print("="*50)
        print(f"\nGlobal Weighted Accuracy: {global_accuracy:.2f}%")
        print(f"  - Editing QA Accuracy: {global_editing_acc:.2f}%")
        print(f"  - Maintain QA Accuracy: {global_maintain_acc:.2f}%")
        print(f"\nCategory-wise Accuracy:")
        for category, stats in sorted(group_accuracies.items()):
            print(f"  {category}: {stats['accuracy']:.2f}% (edit: {stats['editing_accuracy']:.2f}%, maintain: {stats['maintain_accuracy']:.2f}%, n={stats['num_samples']})")

        # Save analysis to JSON for this model
        analysis_data = {
            "model_name": model_name,
            "global_weighted_accuracy": global_accuracy,
            "global_editing_accuracy": global_editing_acc,
            "global_maintain_accuracy": global_maintain_acc,
            "group_accuracies": group_accuracies,
            "total_samples": len(sample_accuracies),
            "total_evaluations": sum(1 for r in evaluation_results if r["model_name"] == model_name),
        }

        analysis_filename = f"{dataset_name}_{model_name}_qwen_analysis.json"
        analysis_file_path = os.path.join(args.output_dir, analysis_filename)

        with open(analysis_file_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=4, ensure_ascii=False)
        print(f"\nAnalysis saved to: {analysis_file_path}")
