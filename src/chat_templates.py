"""
Chat templates for different model families.
"""

# Qwen 2.5 uses the ChatML-style template
QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""
