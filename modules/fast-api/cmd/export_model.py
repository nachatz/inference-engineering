import logging

from app.common.enums.logging import LogFormat
from app.core.config import settings
from app.core.logger import configure_logging
from app.core.logger import logger
from model import seed_everything
from model.config import export_settings
from model.export import export
from model.model import Classifier

configure_logging(LogFormat.TEXT)
# The exporter warns about every torchvision op it skips; this model uses none of them.
logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)

seed_everything(export_settings.seed)
logger.info(export(Classifier(), settings.model.path))
