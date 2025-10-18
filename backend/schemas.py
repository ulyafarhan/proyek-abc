from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    class Config:
        orm_mode = True

class JournalEntryBase(BaseModel):
    content: str

class JournalEntryCreate(JournalEntryBase):
    pass

class JournalEntry(JournalEntryBase):
    id: int
    timestamp: datetime
    analysis_json: str
    owner_id: int
    class Config:
        orm_mode = True