from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from typing import Optional

# Impor dari file-file lokal Anda
from . import models, schemas, database

# --- Konfigurasi Keamanan ---
# PENTING: Di lingkungan produksi, ganti SECRET_KEY ini dengan nilai yang kompleks
# dan simpan sebagai environment variable.
SECRET_KEY = "INI_ADALAH_KUNCI_RAHASIA_YANG_HARUS_ANDA_GANTI"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # Token akan valid selama 60 menit

# Konteks untuk hashing password menggunakan bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Skema OAuth2 yang menunjuk ke endpoint login Anda
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")


# --- Fungsi Utilitas Keamanan ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Memverifikasi password polos dengan hash di database."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Membuat hash dari password polos."""
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    """Membuat JSON Web Token (JWT) baru."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- Fungsi Autentikasi & Dependensi ---

def get_user(db: Session, username: str) -> Optional[models.User]:
    """Mengambil data pengguna dari database berdasarkan username."""
    return db.query(models.User).filter(models.User.username == username).first()

def authenticate_user(db: Session, username: str, password: str) -> Optional[models.User]:
    """Mengautentikasi pengguna. Mengembalikan user object jika valid, None jika tidak."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)) -> models.User:
    """
    Dependensi FastAPI: Mendekode token, memvalidasi, dan mengambil user dari database.
    Ini akan 'disuntikkan' ke setiap endpoint yang memerlukannya.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Tidak dapat memvalidasi kredensial",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: models.User = Depends(get_current_user)) -> models.User:
    """
    Dependensi tambahan untuk mengecek apakah user aktif.
    Untuk saat ini, kita hanya mengembalikan user yang sudah terautentikasi.
    """
    # Jika Anda menambahkan kolom `is_active` di model User, Anda bisa memeriksanya di sini.
    # if not current_user.is_active:
    #     raise HTTPException(status_code=400, detail="User tidak aktif")
    return current_user