from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from backend.database import SessionLocal
from backend.models import DashboardUser
from backend.auth.schemas import (
    LoginRequest,
    LoginResponse,
    RegisterUserRequest,
    UserResponse,
)
from backend.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
)
from backend.auth.dependencies import get_current_user, require_roles


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    db = SessionLocal()

    try:
        user = db.query(DashboardUser).filter(
            DashboardUser.email == payload.email
        ).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled"
            )

        user.last_login_at = datetime.utcnow()
        db.commit()
        db.refresh(user)

        token = create_access_token({
            "sub": user.user_id,
            "email": user.email,
            "role": user.role
        })

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": user
        }

    finally:
        db.close()


@router.get("/me", response_model=UserResponse)
def get_me(current_user: DashboardUser = Depends(get_current_user)):
    return current_user


@router.post(
    "/users",
    response_model=UserResponse,
    dependencies=[Depends(require_roles("admin"))]
)
def create_dashboard_user(payload: RegisterUserRequest):
    db = SessionLocal()

    try:
        existing = db.query(DashboardUser).filter(
            DashboardUser.email == payload.email
        ).first()

        if existing:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )

        user = DashboardUser(
            full_name=payload.full_name,
            email=payload.email,
            password_hash=hash_password(payload.password),
            role=payload.role,
            is_active=True
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return user

    finally:
        db.close()


@router.get(
    "/users",
    response_model=list[UserResponse],
    dependencies=[Depends(require_roles("admin"))]
)
def list_dashboard_users():
    db = SessionLocal()

    try:
        return db.query(DashboardUser).order_by(
            DashboardUser.created_at.desc()
        ).all()

    finally:
        db.close()