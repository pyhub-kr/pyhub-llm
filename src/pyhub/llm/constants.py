"""Constants for LLM providers and models."""

from typing import Dict, List

# Image generation model sizes
IMAGE_GENERATION_SIZES: Dict[str, List[str]] = {
    "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"],
    "dall-e-2": ["256x256", "512x512", "1024x1024"],
}

# Image generation defaults
IMAGE_GENERATION_DEFAULTS: Dict[str, Dict[str, str]] = {
    "dall-e-3": {
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid"
    },
    "dall-e-2": {
        "size": "1024x1024",
        "quality": "standard"
    }
}

# Image generation quality options
IMAGE_GENERATION_QUALITIES: Dict[str, List[str]] = {
    "dall-e-3": ["standard", "hd"],
    "dall-e-2": ["standard"],
}

# Image generation style options
IMAGE_GENERATION_STYLES: Dict[str, List[str]] = {
    "dall-e-3": ["vivid", "natural"],
    "dall-e-2": [],  # DALL-E 2 doesn't support style parameter
}