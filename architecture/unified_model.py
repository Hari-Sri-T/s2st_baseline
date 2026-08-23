import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionLayer(nn.Module):
    """
    The core research contribution: fuses content, speaker, emotion, and prosody.
    Uses Cross-Attention to inject expressive and speaker identity into content tokens.
    """
    def __init__(self, content_dim=512, spk_dim=256, expr_dim=256, fusion_dim=512):
        super().__init__()
        # Project everything to a common fusion dimension
        self.content_proj = nn.Linear(content_dim, fusion_dim)
        self.spk_proj = nn.Linear(spk_dim, fusion_dim)
        self.expr_proj = nn.Linear(expr_dim, fusion_dim)
        
        # We will use MultiHead Attention where Content is the Query, 
        # and Speaker+Expressive are concatenated as Keys/Values.
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=8, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)

    def forward(self, content_tokens, speaker_emb, expr_tokens):
        """
        content_tokens: (Batch, SeqLen, ContentDim)
        speaker_emb: (Batch, 1, SpkDim) or (Batch, SpkDim)
        expr_tokens: (Batch, SeqLen, ExprDim)
        """
        if speaker_emb.dim() == 2:
            speaker_emb = speaker_emb.unsqueeze(1) # (B, 1, SpkDim)
            
        # Project to fusion dim
        q_content = self.content_proj(content_tokens) # (B, S, F)
        k_spk = self.spk_proj(speaker_emb)            # (B, 1, F)
        k_expr = self.expr_proj(expr_tokens)          # (B, S, F)
        
        # Concatenate conditions along the sequence dimension for Keys and Values
        # kv shape: (B, 1 + S, F)
        kv = torch.cat([k_spk, k_expr], dim=1)
        
        # Cross Attention: Inject identity and emotion into the content
        attn_out, _ = self.cross_attn(query=q_content, key=kv, value=kv)
        
        # Add & Norm + FFN
        x = self.layer_norm(q_content + attn_out)
        out = self.ffn(x) + x
        
        return out

class UnifiedExpressiveS2ST(nn.Module):
    """
    The full Unified Speech-to-Speech Architecture proposed in Stage 4.
    """
    def __init__(self, content_dim=512, spk_dim=256, expr_dim=256, fusion_dim=512, num_tgt_langs=22):
        super().__init__()
        
        # 1. Encoders (These would wrap pre-trained models like Whisper/Wav2Vec2 in practice)
        # We define their output dimensions here for the fusion layer to consume.
        self.content_dim = content_dim
        self.spk_dim = spk_dim
        self.expr_dim = expr_dim
        
        # Target language embedding
        self.lang_emb = nn.Embedding(num_tgt_langs, fusion_dim)
        
        # 2. Multimodal Fusion Mechanism
        self.fusion = MultimodalFusionLayer(content_dim, spk_dim, expr_dim, fusion_dim)
        
        # 3. Expressive Decoder Scaffold
        # Takes the fused representation + target language and outputs target acoustic tokens
        self.decoder_rnn = nn.LSTM(fusion_dim * 2, fusion_dim, batch_first=True)
        self.acoustic_head = nn.Linear(fusion_dim, 1024) # Example: to EnCodec vocabulary size

    def forward(self, content_features, speaker_features, expressive_features, target_lang_id):
        """
        content_features: (B, S_src, content_dim)
        speaker_features: (B, spk_dim)
        expressive_features: (B, S_src, expr_dim)
        target_lang_id: (B,)
        """
        # Step 1: Fuse representations (B, S_src, fusion_dim)
        fused_rep = self.fusion(content_features, speaker_features, expressive_features)
        
        # Step 2: Inject Target Language (B, 1, fusion_dim)
        lang_vec = self.lang_emb(target_lang_id).unsqueeze(1)
        
        # Broadcast language embedding to match sequence length (B, S_src, fusion_dim)
        lang_vec_expanded = lang_vec.expand(-1, fused_rep.size(1), -1)
        
        # Concat language with fused representation (B, S_src, fusion_dim * 2)
        decoder_input = torch.cat([fused_rep, lang_vec_expanded], dim=-1)
        
        # Step 3: Decode to target speech representations
        dec_out, _ = self.decoder_rnn(decoder_input) # (B, S_src, fusion_dim)
        acoustic_logits = self.acoustic_head(dec_out) # (B, S_src, vocab_size)
        
        return acoustic_logits
