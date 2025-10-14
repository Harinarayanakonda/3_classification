from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class UserRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    USER: _ClassVar[UserRole]
    ADMIN: _ClassVar[UserRole]

class UserStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PENDING_VERIFICATION: _ClassVar[UserStatus]
    PENDING_APPROVAL: _ClassVar[UserStatus]
    ACTIVE: _ClassVar[UserStatus]
    SUSPENDED: _ClassVar[UserStatus]

class OAuthProvider(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OAUTH_PROVIDER_UNSPECIFIED: _ClassVar[OAuthProvider]
    GOOGLE: _ClassVar[OAuthProvider]
    GITHUB: _ClassVar[OAuthProvider]
    MICROSOFT: _ClassVar[OAuthProvider]
USER: UserRole
ADMIN: UserRole
PENDING_VERIFICATION: UserStatus
PENDING_APPROVAL: UserStatus
ACTIVE: UserStatus
SUSPENDED: UserStatus
OAUTH_PROVIDER_UNSPECIFIED: OAuthProvider
GOOGLE: OAuthProvider
GITHUB: OAuthProvider
MICROSOFT: OAuthProvider

class AuthResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class EmptyRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RegisterUserRequest(_message.Message):
    __slots__ = ("username", "email", "password", "phone_number")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    PHONE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    username: str
    email: str
    password: str
    phone_number: str
    def __init__(self, username: _Optional[str] = ..., email: _Optional[str] = ..., password: _Optional[str] = ..., phone_number: _Optional[str] = ...) -> None: ...

class RegisterAdminRequest(_message.Message):
    __slots__ = ("username", "email", "password", "full_name", "employee_id", "admin_request_reason")
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    EMPLOYEE_ID_FIELD_NUMBER: _ClassVar[int]
    ADMIN_REQUEST_REASON_FIELD_NUMBER: _ClassVar[int]
    username: str
    email: str
    password: str
    full_name: str
    employee_id: str
    admin_request_reason: str
    def __init__(self, username: _Optional[str] = ..., email: _Optional[str] = ..., password: _Optional[str] = ..., full_name: _Optional[str] = ..., employee_id: _Optional[str] = ..., admin_request_reason: _Optional[str] = ...) -> None: ...

class VerifyOTPRequest(_message.Message):
    __slots__ = ("email", "otp_code")
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    OTP_CODE_FIELD_NUMBER: _ClassVar[int]
    email: str
    otp_code: str
    def __init__(self, email: _Optional[str] = ..., otp_code: _Optional[str] = ...) -> None: ...

class LoginRequest(_message.Message):
    __slots__ = ("email", "password")
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    PASSWORD_FIELD_NUMBER: _ClassVar[int]
    email: str
    password: str
    def __init__(self, email: _Optional[str] = ..., password: _Optional[str] = ...) -> None: ...

class LoginResponse(_message.Message):
    __slots__ = ("success", "message", "token", "user_id", "role", "username", "email")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    token: str
    user_id: str
    role: UserRole
    username: str
    email: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., token: _Optional[str] = ..., user_id: _Optional[str] = ..., role: _Optional[_Union[UserRole, str]] = ..., username: _Optional[str] = ..., email: _Optional[str] = ...) -> None: ...

class LoginWithOAuthRequest(_message.Message):
    __slots__ = ("provider", "id_token", "access_token", "email_fallback", "avatar_url", "full_name")
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    ID_TOKEN_FIELD_NUMBER: _ClassVar[int]
    ACCESS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FALLBACK_FIELD_NUMBER: _ClassVar[int]
    AVATAR_URL_FIELD_NUMBER: _ClassVar[int]
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    provider: OAuthProvider
    id_token: str
    access_token: str
    email_fallback: str
    avatar_url: str
    full_name: str
    def __init__(self, provider: _Optional[_Union[OAuthProvider, str]] = ..., id_token: _Optional[str] = ..., access_token: _Optional[str] = ..., email_fallback: _Optional[str] = ..., avatar_url: _Optional[str] = ..., full_name: _Optional[str] = ...) -> None: ...

class CheckUsernameRequest(_message.Message):
    __slots__ = ("username",)
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    username: str
    def __init__(self, username: _Optional[str] = ...) -> None: ...

class CheckUsernameResponse(_message.Message):
    __slots__ = ("is_available",)
    IS_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    is_available: bool
    def __init__(self, is_available: bool = ...) -> None: ...

class ForgotPasswordRequest(_message.Message):
    __slots__ = ("email",)
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    email: str
    def __init__(self, email: _Optional[str] = ...) -> None: ...

class ResetPasswordRequest(_message.Message):
    __slots__ = ("token", "email", "new_password")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    NEW_PASSWORD_FIELD_NUMBER: _ClassVar[int]
    token: str
    email: str
    new_password: str
    def __init__(self, token: _Optional[str] = ..., email: _Optional[str] = ..., new_password: _Optional[str] = ...) -> None: ...

class AdminUser(_message.Message):
    __slots__ = ("user_id", "username", "email", "full_name", "admin_request_reason", "created_at")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    FULL_NAME_FIELD_NUMBER: _ClassVar[int]
    ADMIN_REQUEST_REASON_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    username: str
    email: str
    full_name: str
    admin_request_reason: str
    created_at: str
    def __init__(self, user_id: _Optional[str] = ..., username: _Optional[str] = ..., email: _Optional[str] = ..., full_name: _Optional[str] = ..., admin_request_reason: _Optional[str] = ..., created_at: _Optional[str] = ...) -> None: ...

class AdminListResponse(_message.Message):
    __slots__ = ("admins",)
    ADMINS_FIELD_NUMBER: _ClassVar[int]
    admins: _containers.RepeatedCompositeFieldContainer[AdminUser]
    def __init__(self, admins: _Optional[_Iterable[_Union[AdminUser, _Mapping]]] = ...) -> None: ...

class AdminApprovalRequest(_message.Message):
    __slots__ = ("user_id", "expiry_days")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    EXPIRY_DAYS_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    expiry_days: int
    def __init__(self, user_id: _Optional[str] = ..., expiry_days: _Optional[int] = ...) -> None: ...

class UserActionRequest(_message.Message):
    __slots__ = ("user_id", "reason")
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    user_id: str
    reason: str
    def __init__(self, user_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...
