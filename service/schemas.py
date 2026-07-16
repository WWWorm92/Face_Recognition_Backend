from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    enabled: bool = True
    roi_enabled: bool = False
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_w: float = 1.0
    roi_h: float = 1.0


class SourceUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    enabled: bool | None = None
    roi_enabled: bool | None = None
    roi_x: float | None = None
    roi_y: float | None = None
    roi_w: float | None = None
    roi_h: float | None = None
