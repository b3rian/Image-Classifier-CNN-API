"""Unit tests for the evaluate function in eval.py"""
import pytest
import tensorflow as tf
from unittest.mock import patch, MagicMock
from eval import evaluate

@pytest.fixture
def dummy_config():
    return {
        "model": {"name": "test_model"},
        "training": {"batch_size": 32}
    }

def test_evaluate_function_returns_metrics(dummy_config):
    """Test that the evaluate function returns expected metrics."""
    dummy_metrics = {"loss": 0.1234, "accuracy": 0.9876}

    with patch("eval.get_test_dataset") as mock_get_dataset, \
         patch("eval.tf.keras.models.load_model") as mock_load_model, \
         patch("eval.get_classification_metrics") as mock_get_metrics:

        # Setup mocked test dataset
        mock_test_ds = MagicMock()
        mock_get_dataset.return_value = (_, _, mock_test_ds)

        # Setup mocked model
        mock_model = MagicMock()
        mock_model.evaluate.return_value = dummy_metrics
        mock_load_model.return_value = mock_model

        # Setup dummy metrics
        mock_get_metrics.return_value = ["accuracy"]

        # Call the function under test
        result = evaluate("dummy_model_path", dummy_config)

        # Assertions
        mock_model.compile.assert_called_once()
        mock_model.evaluate.assert_called_once_with(mock_test_ds, return_dict=True)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert result["accuracy"] == dummy_metrics["accuracy"]
