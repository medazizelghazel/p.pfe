from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models
from app.core.dependencies import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.database import get_db
from app.schemas.auth_schema import (
    AuthResponse,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    ProfileUpdateRequest,
    UserResponse,
)
from app.services.email_service import EmailService
from app.utils.password_utils import generate_temporary_password


router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(models.User)
        .filter(models.User.email == payload.email)
        .first()
    )

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
        }
    )

    return AuthResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    user = (
        db.query(models.User)
        .filter(models.User.email == payload.email)
        .first()
    )

    if user is None:
        return {
            "message": "Si cet email existe, un nouveau mot de passe temporaire sera envoyé.",
        }

    temporary_password = generate_temporary_password()

    user.password_hash = hash_password(temporary_password)
    db.commit()

    try:
        EmailService().send_password_reset(
            to_email=user.email,
            full_name=user.full_name,
            temporary_password=temporary_password,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Le mot de passe a été réinitialisé, mais l’email n’a pas été envoyé : {str(e)}",
        )

    return {
        "message": "Si cet email existe, un nouveau mot de passe temporaire sera envoyé.",
    }


@router.get("/me", response_model=UserResponse)
def get_my_profile(
    current_user: models.User = Depends(get_current_user),
):
    return UserResponse.model_validate(current_user)


@router.put("/profile", response_model=UserResponse)
def update_my_profile(
    payload: ProfileUpdateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    existing_email = (
        db.query(models.User)
        .filter(
            models.User.email == payload.email,
            models.User.id != current_user.id,
        )
        .first()
    )

    if existing_email:
        raise HTTPException(
            status_code=400,
            detail="Email already used.",
        )

    current_user.full_name = payload.full_name
    current_user.email = payload.email

    db.commit()
    db.refresh(current_user)

    return UserResponse.model_validate(current_user)


@router.put("/change-password")
def change_my_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect.",
        )

    current_user.password_hash = hash_password(payload.new_password)

    db.commit()

    return {
        "message": "Password changed successfully.",
    }