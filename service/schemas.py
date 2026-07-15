from pydantic import BaseModel, Field


class SourceCreate(BaseModel):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    enabled: bool = True


class SourceUpdate(BaseModel):
    name: str | None = None
    url: str | None = None
    enabled: bool | None = None
