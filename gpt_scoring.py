import os
import json
import argparse
import sys
import time
import base64
import io
import random
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset, Dataset
from PIL import Image
from collections import defaultdict


def encode_image(image_obj):
    """Encode PIL Image object to base64 string."""
    if not isinstance(image_obj, Image.Image):
        return None
    
    # Double the resolution if smaller than 1024x1024
    width, height = image_obj.size
    if width * height < 1024 * 1024 * 0.5:
        new_width = width * 2
        new_height = height * 2
        image_obj = image_obj.resize((new_width, new_height), Image.LANCZOS)
    
    buffered = io.BytesIO()
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")
    image_obj.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_gpt5(prompt: str, image, client: OpenAI):
    """Call GPT-5 API with retry logic."""
    max_attempts = 5
    last_exception = None

    for attempt in range(max_attempts):
        try:
            if image:
                # Handle base64 image input
                if isinstance(image, str) and image.startswith('data:image'):
                    base64_image = image
                else:
                    base64_image = encode_image(image)
                    if not base64_image:
                        raise Exception("Image encoding failed")
                    base64_image = f"data:image/jpeg;base64,{base64_image}"
                
                input_content = [{
                    'role': 'user',
                    'content': [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": base64_image}
                    ]
                }]
            else:
                input_content = prompt

            response = client.responses.create(
                model="gpt-5",
                input=input_content,
                # reasoning={"effort": "minimal"}
            )

            return response.output_text.strip()

        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
            if attempt < max_attempts - 1:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)

    return f"Error: All {max_attempts} attempts failed. Last error: {last_exception}"


def single_qa_evaluation_worker(task_info, client):
    """
    Evaluate a single QA pair by calling GPT-5 to answer and then judge correctness.
    """
    try:
        item_index = task_info["item_index"]
        model_name = task_info["model_name"]
        qa_index = task_info["qa_index"]
        question = task_info["question"]
        ground_truth_answer = task_info["ground_truth_answer"]
        model_image_base64 = task_info["model_image_base64"]

        # Step 1: GPT-5 answers the question based on the image
        qa_prompt = "You are a vision QA assistant. Look at the image carefully and answer the question in the simplest words. Your answer must be the single, definitive, and deterministic response based exclusively on the visual information in the image. Do not infer or add outside information. If the question is not about the image, or the mentioned elements are not visible, return \"N/A\". Directly output the concise answer with no explanation.\nQuestion: "
        model_response = call_gpt5(qa_prompt + question, model_image_base64, client)

        # Step 2: GPT-5 evaluates if the answer is correct
        eval_prompt = (
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
        ).format(question=question, ground_truth_answer=ground_truth_answer, model_response=model_response)

        evaluation_response = call_gpt5(eval_prompt, None, client)
        is_correct = "Correct" in evaluation_response

        return {
            "item_index": item_index,
            "model_name": model_name,
            "qa_index": qa_index,
            "answer": model_response,
            "Correction": is_correct,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
        }

    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {
            "item_index": task_info["item_index"],
            "model_name": task_info["model_name"],
            "qa_index": task_info["qa_index"],
            "error": str(e),
            "answer": "Error occurred",
            "Correction": False,
            "question": task_info["question"],
            "ground_truth_answer": task_info["ground_truth_answer"],
        }


def create_evaluation_tasks(dataset, prefix):
    """
    Create evaluation tasks for each item, model, and QA pair in the dataset.
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

        # Encode all model images to base64
        model_images_base64 = {}
        for model_name in model_names:
            model_image = item.get(f"{prefix}{model_name}")
            if not isinstance(model_image, Image.Image):
                print(f"Warning: Invalid image for model {model_name} in item {item_index}, skipping")
                continue
            
            base64_image = encode_image(model_image)
            if base64_image:
                model_images_base64[model_name] = f"data:image/jpeg;base64,{base64_image}"
            else:
                print(f"Warning: Failed to encode image for model {model_name} in item {item_index}, skipping")

        # Create tasks for each model and QA pair
        for model_name in model_images_base64.keys():
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
                    "model_image_base64": model_images_base64[model_name],
                })

    return tasks


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
        
        if "error" in result:
            results_by_item[item_index][model_name][qa_index]["error"] = result["error"]

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
                
                new_item[f"{model_name}_list"] = ordered_results
                
                # Compute weighted accuracy: 0.1 * maintain_acc + 0.9 * editing_acc
                maintain_acc = (maintain_correct / maintain_total * 100) if maintain_total > 0 else 0.0
                editing_acc = (editing_correct / editing_total * 100) if editing_total > 0 else 0.0
                if maintain_total <= 0:
                    weighted_acc = editing_acc
                else:
                    weighted_acc = 0.1 * maintain_acc + 0.9 * editing_acc
                
                new_item[f"{model_name}_accuracy"] = weighted_acc
                new_item[f"{model_name}_editing_accuracy"] = editing_acc
                new_item[f"{model_name}_maintain_accuracy"] = maintain_acc
        
        final_results.append(new_item)

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-5 Vision QA Evaluation Pipeline")
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
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API Key",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=100,
        help="Number of parallel worker threads",
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

    # Create evaluation tasks
    print("\nCreating evaluation tasks...")
    evaluation_tasks = create_evaluation_tasks(dataset, args.prefix)
    print(f"Total tasks created: {len(evaluation_tasks)}")

    if not evaluation_tasks:
        print("No valid evaluation tasks, exiting")
        sys.exit(1)

    # Initialize OpenAI client
    print("\nInitializing OpenAI client...")
    client = OpenAI(api_key=args.api_key)
    
    # Run parallel evaluation
    print("\nStarting parallel evaluation...")
    evaluation_results = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {
            executor.submit(single_qa_evaluation_worker, task, client): task
            for task in evaluation_tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(evaluation_tasks),
            desc="Evaluating QA pairs",
        ):
            try:
                result = future.result()
                evaluation_results.append(result)
            except Exception as e:
                task = future_to_task[future]
                print(f"\nCritical error processing task: {e}")
                error_result = {
                    "item_index": task["item_index"],
                    "model_name": task["model_name"],
                    "qa_index": task["qa_index"],
                    "error": str(e),
                    "answer": "Error occurred",
                    "Correction": False,
                    "question": task["question"],
                    "ground_truth_answer": task["ground_truth_answer"],
                }
                evaluation_results.append(error_result)

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
            if key.endswith("_accuracy"):
                model_name = key[:-len("_accuracy")]
                all_models.add(model_name)
    
    all_models = sorted(all_models)
    
    if not all_models:
        print("Warning: No models found in results")
        sys.exit(0)

    print(f"\nFound {len(all_models)} model(s): {', '.join(all_models)}")

    # Compute accuracies for each model separately
    dataset_name = os.path.basename(args.dataset_path.strip("/"))
    
    for model_name in all_models:
        accuracy_key = f"{model_name}_accuracy"
        editing_key = f"{model_name}_editing_accuracy"
        maintain_key = f"{model_name}_maintain_accuracy"
        
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

        analysis_filename = f"{dataset_name}_{model_name}_analysis.json"
        analysis_file_path = os.path.join(args.output_dir, analysis_filename)

        with open(analysis_file_path, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=4, ensure_ascii=False)
        print(f"\nAnalysis saved to: {analysis_file_path}")