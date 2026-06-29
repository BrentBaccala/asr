"""Process-wide NLLB-200 translator (CTranslate2 int8, CPU).

A single instance is shared by every Session — the model is expensive to
load (~1 GB resident) and stateless across calls, so per-session copies
would only waste memory. `translate_multi` batches all targets through
one `translate_batch` (shared encoder pass) exactly as freesoft-asr does.

ctranslate2 is blocking C++; callers in asyncio context should run
`translate_multi` in a thread executor.
"""
import threading


class NllbTranslator:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls, nllb_dir: str, beam: int = 1) -> "NllbTranslator":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(nllb_dir, beam)
            return cls._instance

    def __init__(self, nllb_dir: str, beam: int = 1):
        import ctranslate2
        from transformers import AutoTokenizer
        self.beam = beam
        self._tok = AutoTokenizer.from_pretrained(nllb_dir)
        self._tr = ctranslate2.Translator(
            nllb_dir, device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=8)
        # translate_batch must not be entered concurrently from threads.
        self._lock = threading.Lock()

    def translate_multi(self, text: str, targets: list,
                        src_lang: str | None = None) -> dict:
        """Translate `text` into each FLORES-200 code in `targets`.
        Returns {code -> translated text}. A target equal to the source
        is passed through unchanged (NLLB X->X paraphrases). Empty input
        or empty target list -> {}."""
        targets = [t for t in targets if t]
        if not text.strip() or not targets:
            return {}
        with self._lock:
            if src_lang:
                self._tok.src_lang = src_lang
            src = self._tok.convert_ids_to_tokens(self._tok.encode(text))
            batch_src, batch_tgt, need = [], [], []
            out = {}
            for t in targets:
                if src_lang and t == src_lang:
                    out[t] = text.strip()
                    continue
                batch_src.append(src)
                batch_tgt.append([t])
                need.append(t)
            if need:
                r = self._tr.translate_batch(
                    batch_src, target_prefix=batch_tgt,
                    beam_size=self.beam, max_decoding_length=512)
                for t, res in zip(need, r):
                    hyp = res.hypotheses[0]
                    if hyp and hyp[0] == t:
                        hyp = hyp[1:]
                    out[t] = self._tok.decode(
                        self._tok.convert_tokens_to_ids(hyp)).strip()
        return {t: out[t] for t in targets if t in out}
