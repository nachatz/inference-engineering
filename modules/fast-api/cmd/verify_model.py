import sys

from app.common.enums.logging import LogFormat
from app.core.config import settings
from app.core.logger import configure_logging
from app.core.logger import logger
from model import seed_everything
from model.config import export_settings
from model.model import Classifier
from model.verify import representative_inputs
from model.verify import verify

configure_logging(LogFormat.TEXT)

seed_everything(export_settings.seed)
model = Classifier()

batch_sizes = sorted({*export_settings.verify_batch_sizes, settings.model.max_batch})
inputs = representative_inputs(model.d_in, batch_sizes)
verified = verify(model, settings.model.path, inputs, export_settings.tolerance)
if not verified.passed:
    logger.error(verified)
    sys.exit(1)

logger.info(verified)
