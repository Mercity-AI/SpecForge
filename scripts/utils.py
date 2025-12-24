from huggingface_hub import login

# Hugging Face authentication
HF_TOKEN_RISHIK66 = "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"

def login_to_hf():
    """Login to Hugging Face Hub."""
    login(token=HF_TOKEN_RISHIK66)
    print("[INFO] Logged in to Hugging Face successfully!")