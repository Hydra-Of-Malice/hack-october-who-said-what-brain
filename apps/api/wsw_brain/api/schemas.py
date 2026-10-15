"""Request bodies (responses are the plain dict shapes documented in wsw_brain/contracts.py)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean(value: str) -> str:
    return " ".join(value.split())


class CreateMeeting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field("", max_length=200)

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        return _clean(value)


class NameSpeaker(BaseModel):
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        cleaned = _clean(value)
        if not cleaned:
            raise ValueError("name must not be blank")
        return cleaned


class SetActionStatus(BaseModel):
    status: Literal["open", "done"]


class SignIn(BaseModel):
    token: str = Field(min_length=1, max_length=512)


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2000)
    meeting_id: str | None = Field(None, max_length=64)

    @field_validator("session_id", "question")
    @classmethod
    def _required(cls, value: str) -> str:
        cleaned = _clean(value)
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("meeting_id")
    @classmethod
    def _meeting(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None
