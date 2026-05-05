import os
import time

from sqlalchemy.exc import OperationalError

from backend.database import Base, engine, SessionLocal
from backend.models import DashboardUser
from backend.auth.security import hash_password


def seed_default_admin():
    """
    Create the first admin user automatically when the backend starts.

    This is safe to run on every container startup:
    - If the admin does not exist, it creates it.
    - If the admin already exists, it does nothing.
    - It only resets the password if RESET_DEFAULT_ADMIN_PASSWORD=true.
    """

    create_default_admin = os.getenv("CREATE_DEFAULT_ADMIN", "true").lower()

    if create_default_admin not in ("true", "1", "yes"):
        print("[BOOTSTRAP] Default admin creation disabled.", flush=True)
        return

    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")
    admin_name = os.getenv("DEFAULT_ADMIN_NAME", "System Admin")
    reset_password = os.getenv(
        "RESET_DEFAULT_ADMIN_PASSWORD",
        "false"
    ).lower() in ("true", "1", "yes")

    max_retries = int(os.getenv("DB_BOOTSTRAP_RETRIES", "10"))
    retry_delay = float(os.getenv("DB_BOOTSTRAP_RETRY_DELAY", "2"))

    for attempt in range(1, max_retries + 1):
        db = None

        try:
            Base.metadata.create_all(bind=engine)

            db = SessionLocal()

            existing = db.query(DashboardUser).filter(
                DashboardUser.email == admin_email
            ).first()

            if existing:
                if reset_password:
                    existing.password_hash = hash_password(admin_password)
                    existing.role = "admin"
                    existing.is_active = True
                    db.commit()

                    print(
                        "[BOOTSTRAP] Existing admin password reset.",
                        flush=True,
                    )
                else:
                    print(
                        f"[BOOTSTRAP] Admin already exists: {admin_email}",
                        flush=True,
                    )

                return

            admin = DashboardUser(
                full_name=admin_name,
                email=admin_email,
                password_hash=hash_password(admin_password),
                role="admin",
                is_active=True,
            )

            db.add(admin)
            db.commit()

            print("[BOOTSTRAP] Default admin created successfully.", flush=True)
            print(f"[BOOTSTRAP] Email: {admin_email}", flush=True)

            return

        except OperationalError as exc:
            print(
                f"[BOOTSTRAP] Database not ready. "
                f"Attempt {attempt}/{max_retries}. Error: {exc}",
                flush=True,
            )

            time.sleep(retry_delay)

        except Exception as exc:
            print(f"[BOOTSTRAP] Failed to create default admin: {exc}", flush=True)
            raise

        finally:
            if db:
                db.close()

    raise RuntimeError(
        "[BOOTSTRAP] Could not connect to database to create default admin."
    )