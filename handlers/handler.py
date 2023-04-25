"""Handler for the opus-mt models."""

import logging

from transformers import AutoTokenizer, pipeline
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class TransformersClassifierHandler(BaseHandler):
    """Handler class for opus-mt models."""

    def __init__(self):
        """Initialize class."""
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """Load the hugging face pipeline."""
        model_dir = ctx.system_properties.get("model_dir")

        self.hf_pipeline = pipeline(
            "translation",
            model=model_dir,
            tokenizer=AutoTokenizer.from_pretrained(
                model_dir, truncation=True, padding=False
            ),
            truncation=True,
        )

        self.initialized = True

    def preprocess(self, data):
        """Take the column we want to predict."""
        data = [datapoint[1] for datapoint in data]
        return data

    def inference(self, inputs):
        """Predict the class of a text using a trained transformer model."""
        return self.hf_pipeline(inputs)

    def postprocess(self, inference_output):
        """Convert the output of the model to a list of translations."""
        clean_preds = [
            pred["translation_text"]
            if type(pred) != list
            else pred[0]["translation_text"]
            for pred in inference_output
        ]
        return clean_preds
