from pydantic import BaseModel, EmailStr
from typing import Literal


UserRole = Literal["admin", "da", "expert"]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterUserRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: UserRole


class UserResponse(BaseModel):
    user_id: str
    full_name: str
    email: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse