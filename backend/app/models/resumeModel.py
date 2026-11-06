from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ResumeData(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name : str | None = None
    email : EmailStr | None = None
    phone : str | None = None
    skills : list[str] = Field(default_factory=list)