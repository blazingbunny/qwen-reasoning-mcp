import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

# Set HF Token from environment for faster/unauthenticated-free downloads
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("HF_TOKEN found, logging in...")
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in environment. Downloads may be slow or rate-limited.")

# Load a tiny but capable model for reasoning
# Qwen2.5-Coder-0.5B is optimized for code and logic, fitting well in HF Free Tier RAM
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def generate_reasoning(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate reasoning or code logic based on a prompt.
    
    Args:
        prompt: The instructions or query for the model.
        max_new_tokens: Maximum length of the response.
    """
    messages = [
        {"role": "system", "content": "You are a concise reasoning engine for a memory agent. Extract facts and logic precisely."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3, # Lower temperature for more stable reasoning
            top_p=0.9
        )
    
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# Create Gradio UI
with gr.Blocks(title="Qwen Reasoning MCP") as demo:
    gr.Markdown("""
    # 🧠 Qwen Reasoning MCP Server
    
    **FREE, Self-hosted Reasoning** - Powered by Qwen2.5-Coder-0.5B.
    
    This Space provides a lightweight reasoning engine for AI agents. 
    The `generate_reasoning` function is automatically exposed as an MCP tool.
    """)
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="Enter reasoning task...", lines=5)
            max_tokens = gr.Slider(128, 2048, value=512, step=128, label="Max Tokens")
            submit_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Response", lines=10)
            
    submit_btn.click(fn=generate_reasoning, inputs=[prompt_input, max_tokens], outputs=output_text)

# Launch as MCP Server
if __name__ == "__main__":
    # Gradio handles the MCP protocol mapping automatically
    demo.launch(mcp_server=True)
