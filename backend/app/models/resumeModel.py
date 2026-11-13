from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ResumeData(BaseModel):
    model_config = ConfigDict(extra='forbid', revalidate_instances="always")

    name : str | None = Field(default=None , description='Original name of Person')
    email : EmailStr | None = Field(default=None)
    phone : str | None = Field(default=None , description='Phone number of a person like 91 9876543210')
    skills : list[str] = Field(default_factory=list)