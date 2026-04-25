from backend.config import (
    CACHE_DIR,
    CONFIG_PATH,
    LLAMA_SHORT_FLAGS,
    LLAMA_SWAP_CONTAINER,
    QUANT_REGEX,
    SERVED_DIR,
    SHA_RE,
)
import re


class TestConstants:
    def test_cache_dir(self):
        assert CACHE_DIR == "/models/.cache"

    def test_served_dir(self):
        assert SERVED_DIR == "/models/served"

    def test_config_path(self):
        assert CONFIG_PATH == "/models/served/config.yaml"

    def test_state_file(self):
        from backend.config import STATE_FILE
        assert STATE_FILE == "/models/served/state.json"

    def test_llama_swap_container_default(self):
        assert LLAMA_SWAP_CONTAINER == "llama-swap"


class TestQuantRegex:
    def test_matches_q4_k_m(self):
        match = re.search(QUANT_REGEX, "model-Q4_K_M.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "Q4_K_M"

    def test_matches_ud_iq4_xs(self):
        match = re.search(QUANT_REGEX, "model-UD-IQ4_XS.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "UD-IQ4_XS"

    def test_matches_bf16(self):
        match = re.search(QUANT_REGEX, "model-BF16.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "BF16"

    def test_matches_f16(self):
        match = re.search(QUANT_REGEX, "model-F16.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "F16"

    def test_matches_f32(self):
        match = re.search(QUANT_REGEX, "model-F32.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "F32"

    def test_matches_sharded(self):
        match = re.search(QUANT_REGEX, "model-Q4_K_M-00001-of-00002.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "Q4_K_M"

    def test_matches_mmproj(self):
        match = re.search(QUANT_REGEX, "mmproj-F16.gguf", re.IGNORECASE)
        assert match is not None
        assert match.group(1) == "F16"

    def test_no_match_non_gguf(self):
        match = re.search(QUANT_REGEX, "model.txt", re.IGNORECASE)
        assert match is None

    def test_no_match_invalid_quant(self):
        match = re.search(QUANT_REGEX, "model-INVALID.gguf", re.IGNORECASE)
        assert match is None


class TestSHARegex:
    def test_matches_full_sha(self):
        path = "/snapshots/abc123def456789012345678901234567890abcd/"
        match = SHA_RE.search(path)
        assert match is not None
        assert match.group(1) == "abc123def456789012345678901234567890abcd"

    def test_no_match_short_sha(self):
        path = "/snapshots/abc123/"
        match = SHA_RE.search(path)
        assert match is None


class TestLlamaShortFlags:
    def test_contains_ngl(self):
        assert "ngl" in LLAMA_SHORT_FLAGS

    def test_contains_ctx_size(self):
        assert "ctx_size" not in LLAMA_SHORT_FLAGS

    def test_contains_host(self):
        assert "host" not in LLAMA_SHORT_FLAGS

    def test_contains_single_char_flags(self):
        assert "a" in LLAMA_SHORT_FLAGS
        assert "b" in LLAMA_SHORT_FLAGS
        assert "h" in LLAMA_SHORT_FLAGS
