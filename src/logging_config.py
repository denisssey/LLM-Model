import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


for noise in ("httpx", "urllib3", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noise).setLevel(logging.WARNING)
