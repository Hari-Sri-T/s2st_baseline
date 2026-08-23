"""
Tests the Stage 4 Unified Architecture with dummy tensors to ensure shape compatibility and compilation.
"""
import torch
from architecture.unified_model import UnifiedExpressiveS2ST

def main():
    print("Initializing UnifiedExpressiveS2ST Architecture...")
    
    # Model dimensions
    batch_size = 4
    seq_len = 120
    
    content_dim = 512
    spk_dim = 256
    expr_dim = 256
    fusion_dim = 512
    num_langs = 22
    
    model = UnifiedExpressiveS2ST(
        content_dim=content_dim,
        spk_dim=spk_dim,
        expr_dim=expr_dim,
        fusion_dim=fusion_dim,
        num_tgt_langs=num_langs
    )
    
    print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")
    
    # Create dummy embeddings representing the outputs of our frozen pre-trained encoders
    print(f"Creating dummy inputs (Batch={batch_size}, SeqLen={seq_len})...")
    
    # 1. Content (e.g. Whisper representations)
    dummy_content = torch.randn(batch_size, seq_len, content_dim)
    
    # 2. Speaker (e.g. ECAPA-TDNN / x-vector, single vector per speaker)
    dummy_speaker = torch.randn(batch_size, spk_dim)
    
    # 3. Expressive (e.g. Wav2Vec2 emotion embeddings + F0)
    dummy_expressive = torch.randn(batch_size, seq_len, expr_dim)
    
    # 4. Target Language (e.g. 0=Hindi, 1=Telugu, etc.)
    dummy_target_lang = torch.randint(0, num_langs, (batch_size,))
    
    # Run the forward pass!
    print("Running forward pass through MultimodalFusionLayer and ExpressiveDecoder...")
    output_logits = model(dummy_content, dummy_speaker, dummy_expressive, dummy_target_lang)
    
    print(f"Success! Output acoustic logits shape: {output_logits.shape}")
    
    # Check expected shape (Batch, SeqLen, VocabSize)
    assert output_logits.shape == (batch_size, seq_len, 1024), "Output shape mismatch!"
    print("Shape verification passed.")

if __name__ == "__main__":
    main()
