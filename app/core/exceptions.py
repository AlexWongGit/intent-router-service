from __future__ import annotations


class AppException(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = "app_error",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class BadRequestException(AppException):
    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(
            message,
            error_code="bad_request",
            status_code=400,
        )


class ModelLoadException(AppException):
    def __init__(self, message: str = "Model load failed") -> None:
        super().__init__(
            message,
            error_code="model_load_failed",
            status_code=500,
        )


class PredictionException(AppException):
    def __init__(self, message: str = "Prediction failed") -> None:
        super().__init__(
            message,
            error_code="prediction_failed",
            status_code=500,
        )