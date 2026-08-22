"""
Translation wrapper around AI4Bharat's IndicTrans2.
Roadmap doc, Stage 1 / Baseline A translation candidate.

Requires: pip install indictranstoolkit  (AI4Bharat's official preprocessing toolkit)
          pip install transformers accelerate
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:
    IndicProcessor = None
    print(
        "[WARN] IndicTransToolkit not found. Install with:\n"
        "  pip install git+https://github.com/VarunGumma/IndicTransToolkit.git\n"
        "Sentence normalization/preprocessing will be skipped (lower quality)."
    )


def _pick_checkpoint(src_flores: str, tgt_flores: str, checkpoints: dict) -> str:
    """Pick the right IndicTrans2 checkpoint for the src->tgt direction."""
    src_is_en = src_flores == "eng_Latn"
    tgt_is_en = tgt_flores == "eng_Latn"
    if src_is_en and not tgt_is_en:
        return checkpoints["en-indic"]
    if not src_is_en and tgt_is_en:
        return checkpoints["indic-en"]
    if not src_is_en and not tgt_is_en:
        return checkpoints["indic-indic"]
    raise ValueError("English-to-English translation requested; nothing to do.")


class IndicTrans2Translator:
    def __init__(self, checkpoints: dict, device: str = "cuda"):
        self.checkpoints = checkpoints
        self.device = device
        self._loaded = {}  # cache: checkpoint_name -> (model, tokenizer)
        self.processor = IndicProcessor(inference=True) if IndicProcessor else None

    def _load(self, checkpoint: str):
        if checkpoint not in self._loaded:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint, trust_remote_code=True, torch_dtype=torch.float16
            ).to(self.device)
            model.eval()
            self._loaded[checkpoint] = (model, tokenizer)
        return self._loaded[checkpoint]

    def translate(self, text: str, src_flores: str, tgt_flores: str) -> str:
        """
        Args:
            text: source sentence(s)
            src_flores: FLORES-200 code, e.g. "tel_Telu"
            tgt_flores: FLORES-200 code, e.g. "hin_Deva"
        Returns:
            translated text (str)
        """
        checkpoint = _pick_checkpoint(src_flores, tgt_flores, self.checkpoints)
        model, tokenizer = self._load(checkpoint)

        if self.processor:
            batch = self.processor.preprocess_batch([text], src_lang=src_flores, tgt_lang=tgt_flores)
        else:
            batch = [text]

        inputs = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

        if self.processor:
            decoded = self.processor.postprocess_batch(decoded, lang=tgt_flores)

        return decoded[0]
