from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import jwt
from passlib.context import CryptContext
import secrets
import os
from app import models, schemas
from app.database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token handling
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> models.User:
    """Get the current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    
    return user

def get_current_active_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    """Get the current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(allowed_roles: List[str]):
    """Dependency to require specific user roles"""
    def role_checker(current_user: models.User = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return current_user
    return role_checker

def require_permission(required_permission: str):
    """Dependency to require specific permission"""
    def permission_checker(current_user: models.User = Depends(get_current_active_user)):
        # Check if user has the required permission
        user_permissions = []
        
        # Get role-based permissions
        if current_user.role in schemas.ROLE_PERMISSIONS:
            user_permissions.extend(schemas.ROLE_PERMISSIONS[current_user.role])
        
        # Check if user has the required permission
        if required_permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {required_permission}"
            )
        return current_user
    return permission_checker

# -------------------- AUTHENTICATION ENDPOINTS --------------------

@router.post("/login", response_model=schemas.Token)
def login(user_credentials: schemas.UserLogin, request: Request, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    user = db.query(models.User).filter(models.User.username == user_credentials.username).first()
    
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        # Log failed attempt
        if user:
            user.login_attempts += 1
            if user.login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=30)
            db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is locked
    if user.locked_until and user.locked_until > datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to multiple failed login attempts"
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )
    
    # Reset login attempts on successful login
    user.login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Get user permissions
    user_permissions = schemas.ROLE_PERMISSIONS.get(user.role, [])
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": str(user.user_id), "role": user.role},
        expires_delta=access_token_expires
    )
    
    # Log successful login
    audit_log = models.AuditLog(
        user_id=user.user_id,
        action="LOGIN",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        description="User logged in successfully",
        status="SUCCESS"
    )
    db.add(audit_log)
    db.commit()
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Return in seconds
        "user_info": {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "permissions": user_permissions,
            "dashboard_config": schemas.ROLE_DASHBOARD_CONFIG.get(user.role, {})
        }
    }

@router.post("/logout")
def logout(
    current_user: models.User = Depends(get_current_active_user),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Log out the current user"""
    # Log logout
    audit_log = models.AuditLog(
        user_id=current_user.user_id,
        action="LOGOUT",
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
        description="User logged out",
        status="SUCCESS"
    )
    db.add(audit_log)
    db.commit()
    
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=schemas.CurrentUser)
def get_current_user_info(current_user: models.User = Depends(get_current_active_user)):
    """Get current user information"""
    user_permissions = schemas.ROLE_PERMISSIONS.get(current_user.role, [])
    
    return schemas.CurrentUser(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role,
        permissions=user_permissions,
        is_active=current_user.is_active,
        organization=current_user.organization
    )

@router.post("/change-password")
def change_password(
    password_change: schemas.PasswordChange,
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password"""
    # Verify current password
    if not verify_password(password_change.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_change.new_password)
    db.commit()
    
    # Log password change
    audit_log = models.AuditLog(
        user_id=current_user.user_id,
        action="CHANGE_PASSWORD",
        description="User changed password",
        status="SUCCESS"
    )
    db.add(audit_log)
    db.commit()
    
    return {"message": "Password changed successfully"}

@router.post("/forgot-password")
def forgot_password(password_reset: schemas.PasswordReset, db: Session = Depends(get_db)):
    """Request password reset"""
    user = db.query(models.User).filter(models.User.email == password_reset.email).first()
    
    if user:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        user.reset_token = reset_token
        user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
        db.commit()
        
        # In a real application, send email with reset token
        # For now, return the token (remove this in production!)
        return {"message": "Password reset token generated", "reset_token": reset_token}
    
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent"}

@router.post("/reset-password")
def reset_password(password_reset: schemas.PasswordResetConfirm, db: Session = Depends(get_db)):
    """Reset password using token"""
    user = db.query(models.User).filter(
        models.User.reset_token == password_reset.token
    ).first()
    
    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Update password
    user.hashed_password = get_password_hash(password_reset.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    user.login_attempts = 0  # Reset failed attempts
    user.locked_until = None
    db.commit()
    
    return {"message": "Password reset successfully"}

# -------------------- USER MANAGEMENT (Admin only) --------------------

@router.post("/users", response_model=schemas.UserOut)
def create_user(
    user: schemas.UserCreate,
    current_user: models.User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db)
):
    """Create a new user (Admin only)"""
    # Check if username already exists
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(models.User).filter(models.User.email == user.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_dict = user.dict(exclude={"password"})
    new_user = models.User(
        **user_dict,
        hashed_password=get_password_hash(user.password),
        created_by=current_user.user_id
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.get("/users", response_model=List[schemas.UserOut])
def get_users(
    skip: int = 0,
    limit: int = 100,
    role: Optional[str] = None,
    current_user: models.User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db)
):
    """Get all users (Admin only)"""
    query = db.query(models.User)
    
    if role:
        query = query.filter(models.User.role == role)
    
    users = query.offset(skip).limit(limit).all()
    return users

@router.get("/users/{user_id}", response_model=schemas.UserOut)
def get_user(
    user_id: UUID,
    current_user: models.User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db)
):
    """Get a specific user (Admin only)"""
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=schemas.UserOut)
def update_user(
    user_id: UUID,
    user_update: schemas.UserUpdate,
    current_user: models.User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db)
):
    """Update a user (Admin only)"""
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    return user

@router.delete("/users/{user_id}")
def deactivate_user(
    user_id: UUID,
    current_user: models.User = Depends(require_role(["ADMIN"])),
    db: Session = Depends(get_db)
):
    """Deactivate a user (Admin only)"""
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Don't allow deactivating yourself
    if user.user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    user.is_active = False
    db.commit()
    
    return {"message": "User deactivated successfully"}

# -------------------- ROLE-BASED ACCESS HELPERS --------------------

@router.get("/check-permission/{permission}")
def check_permission(
    permission: str,
    current_user: models.User = Depends(get_current_active_user)
):
    """Check if current user has a specific permission"""
    user_permissions = schemas.ROLE_PERMISSIONS.get(current_user.role, [])
    has_permission = permission in user_permissions
    
    return {
        "permission": permission,
        "has_permission": has_permission,
        "user_role": current_user.role,
        "all_permissions": user_permissions
    }

@router.get("/dashboard-config")
def get_dashboard_config(current_user: models.User = Depends(get_current_active_user)):
    """Get dashboard configuration for current user role"""
    config = schemas.ROLE_DASHBOARD_CONFIG.get(current_user.role, {})
    user_permissions = schemas.ROLE_PERMISSIONS.get(current_user.role, [])
    
    return {
        "role": current_user.role,
        "user_info": {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "full_name": current_user.full_name,
            "organization": current_user.organization,
            "department": current_user.department
        },
        "dashboard_config": config,
        "permissions": user_permissions
    }