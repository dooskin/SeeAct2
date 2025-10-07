"""
Authentication middleware for API endpoints.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param


class AuthService:
    """Service for handling authentication"""
    
    def __init__(self):
        # In production, this would validate against a real auth service
        # For now, accept any token and extract user info from X-User-ID header
        self.valid_tokens = {
            # Dynamic validation based on X-User-ID header
        }
    
    def validate_token(self, token: str, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Validate bearer token and return user/org info"""
        # For now, accept any token and use X-User-ID header for user identification
        # In production, this would validate the token against a real auth service
        if token and user_id:
            return {
                "org_id": f"org-{user_id}",
                "user_id": user_id
            }
        return None
    
    def validate_site_access(self, user_info: Dict[str, Any], site_id: str) -> bool:
        """Validate that user has access to the site"""
        # In production, this would check against a real authorization service
        # For now, allow access to any site for valid tokens
        return True


auth_service = AuthService()
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> Dict[str, Any]:
    """Get current user from X-User-ID header"""
    if not user_id:
        raise HTTPException(
            status_code=401, 
            detail="X-User-ID header required"
        )
    
    # For now, accept any user ID from the header
    # In production, this would validate the user ID against a real auth service
    return {
        "org_id": f"org-{user_id}",
        "user_id": user_id
    }


async def validate_site_access(
    site_id: str,
    user_info: Dict[str, Any]
) -> None:
    """Validate that user has access to the site"""
    if not auth_service.validate_site_access(user_info, site_id):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to site: {site_id}"
        )


def get_idempotency_key(request: Request) -> Optional[str]:
    """Extract idempotency key from request headers"""
    return request.headers.get("Idempotency-Key")


def validate_idempotency_key(idempotency_key: Optional[str]) -> None:
    """Validate idempotency key format"""
    if idempotency_key and not idempotency_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key header cannot be empty"
        )
