"""
Chat templates for different model families.

Each model family has its own chat format that must be used exactly as
specified during training. Using the wrong template will result in
degraded model performance or nonsensical outputs.
"""

# Qwen 2.5 uses the ChatML-style template
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# Llama 3 template (for reference/comparison)
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Phi-3 template (for reference)
PHI3_CHAT_TEMPLATE = """
