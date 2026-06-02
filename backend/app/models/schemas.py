from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted resale price in USD")
    confidence_low: float = Field(..., description="Lower bound of approximate confidence range")
    confidence_high: float = Field(..., description="Upper bound of approximate confidence range")
    input_summary: InputSummary
    warnings: list[str] = Field(default_factory=list)


class InputSummary(BaseModel):
    description_length: int = Field(default=0)
    brand: str = Field(default="")
    brand_known: bool = Field(default=True)
    category: str = Field(default="")
    category_known: bool = Field(default=True)
    condition: str = Field(default="")
    image_provided: bool = Field(default=False)
    product_name: str = Field(default="")


class BrandItem(BaseModel):
    brand: str


class CategoryItem(BaseModel):
    category: str


class ConditionItem(BaseModel):
    condition_id: int
    label: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    image_agent_loaded: bool
    model_files_present: list[str]


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
