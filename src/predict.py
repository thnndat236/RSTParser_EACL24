import json
import os
import torch
from datetime import datetime
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from parse.parse import parse_dataset

class Config(Tap):
    # Base model name on HF
    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    
    # LoRA Adapter IDs from Hugging Face
    span_lora_params: str | None = None
    nuc_lora_params: str | None = None
    rel_lora_params: str | None = None
    nuc_rel_lora_params: str | None = None
    rel_with_nuc_lora_params: str | None = None
    top_down_lora_params: str | None = None

    # Inference settings
    parse_type: str = "bottom_up" 
    rel_type: str = "rel_with_nuc"
    save_result_dir: str = "results/hf_inference"
    save_dir_name: str | None = None
    zero_shot: bool = False
    corpus: str = "rstdt" 

# --- Giữ nguyên các hàm helper từ code cũ của bạn ---
def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def load_model(config, model_type_list):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tokenizer, model)

    peft_model = None
    for model_type in model_type_list:
        params_path = getattr(config, f"{model_type}_lora_params")
        if peft_model is None:
            # PeftModel.from_pretrained sẽ tự động tải từ HF nếu params_path là Model ID
            peft_model = PeftModel.from_pretrained(model, params_path, adapter_name=model_type)
        else:
            peft_model.load_adapter(params_path, model_type)
    peft_model.eval()
    return peft_model, tokenizer

def get_model_type_list(config):
    model_type_list = ["span"] if config.parse_type == "bottom_up" else ["top_down"]
    if config.rel_type == "rel": model_type_list += ["nuc", "rel"]
    elif config.rel_type == "rel_with_nuc": model_type_list += ["nuc", "rel_with_nuc"]
    elif config.rel_type == "nuc_rel": model_type_list += ["nuc_rel"]
    return model_type_list

def run_inference(config):
    # Khởi tạo kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.save_result_dir, config.save_dir_name or timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Load Model
    model_types = get_model_type_list(config)
    model, tokenizer = load_model(config, model_types)

    # Data thử nghiệm
    my_edus = [
        "Westinghouse Electric Corp. said",
        "it will buy Shaw-Walker Co.",
        "Terms weren't disclosed.",
        "Shaw-Walker,",
        "based in Muskegon, Mich.,",
        "makes metal files and desks, and seating and office systems furniture."
    ]
    
    input_data = [{"doc_id": "hf_test_sample", "edu_strings": my_edus, "rst_tree": ""}]

    print("--- Đang phân tích RST Tree bằng mô hình từ Hugging Face ---")
    with torch.no_grad():
        output = parse_dataset(
            input_data, model, tokenizer,
            parse_type=config.parse_type, rel_type=config.rel_type, corpus=config.corpus
        )

    # Lưu kết quả
    tree_result = output["pred_tree"][0]
    with open(os.path.join(save_path, "result.tree"), "w") as f:
        f.write(str(tree_result))
    
    print(f"\n[Xong] Cây RST dự đoán:\n{tree_result}")
    print(f"Lưu tại: {save_path}/result.tree")

if __name__ == "__main__":
    config = Config().parse_args()
    run_inference(config)