from __future__ import annotations

from transformers import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Gemma4AudioConfig(PretrainedConfig):
    model_type = "gemma4_audio"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Gemma4VisionConfig(PretrainedConfig):
    model_type = "gemma4_vision"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "audio_config": Gemma4AudioConfig,
        "vision_config": Gemma4VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_config = self._coerce_sub_config(text_config, Gemma4TextConfig)
        self.audio_config = self._coerce_sub_config(audio_config, Gemma4AudioConfig)
        self.vision_config = self._coerce_sub_config(vision_config, Gemma4VisionConfig)

    @staticmethod
    def _coerce_sub_config(value, cls):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, PretrainedConfig):
            return cls(**value.to_dict())
        if isinstance(value, dict):
            return cls(**value)
        return value
