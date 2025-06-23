from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from reworkd_platform.schemas.agent import LLM_MODEL_MAX_TOKENS, LLM_Model
from reworkd_platform.schemas.user import UserBase
from reworkd_platform.settings import Settings, settings as global_settings
from reworkd_platform.web.api.dependencies import get_current_user

router = APIRouter()


class ModelWithAccess(BaseModel):
    name: str
    max_tokens: int
    has_access: bool = Field(
        default=False, description="Whether the user has access to this model"
    )

    @staticmethod
    def from_model( # This method is not strictly needed anymore if has_access is determined prior to calling
        name: str, max_tokens: int, determined_has_access: bool
    ) -> "ModelWithAccess":
        # The 'has_access' is now determined by the caller loop logic in get_models
        return ModelWithAccess(name=name, max_tokens=max_tokens, has_access=determined_has_access)


@router.get("")
async def get_models(
    user: UserBase = Depends(get_current_user),
    settings: Settings = Depends(lambda: global_settings),
) -> List[ModelWithAccess]:
    available_models = []
    for model_name_literal, tokens in LLM_MODEL_MAX_TOKENS.items():
        # Ensure we are working with the string value of the Literal
        model_name = str(model_name_literal)

        is_ollama_model = model_name.startswith("ollama/")
        is_openai_model = model_name.startswith("gpt-")

        access_granted = False
        if is_ollama_model:
            if settings.ollama_enabled:
                access_granted = True
        elif is_openai_model:
            # OpenAI models are accessible if:
            # 1. User is logged in (platform may use its own keys or user provides custom key via UI).
            # 2. Platform has a valid OpenAI API key configured (not the placeholder).
            # 3. Platform has a specific OpenAI API base configured (e.g., Azure, other proxy).
            # The user providing a custom key via UI is handled by the UI allowing model selection;
            # this endpoint primarily determines which models the *platform* can support.
            platform_has_valid_openai_key = settings.openai_api_key and settings.openai_api_key != "<Should be updated via env>"
            if user or platform_has_valid_openai_key or settings.openai_api_base:
                access_granted = True

        # The from_model static method's has_access logic also needs to be in sync.
        # We're calling from_model only for models where access_granted is true based on the loop's logic.
        # So, the has_access inside from_model will re-evaluate but should yield the same result
        # if its own logic is consistent.
        # For clarity, we can simplify `ModelWithAccess.from_model` as its `has_access` is determined here.

        if access_granted:
            # Pass the already determined access_granted status
            model_instance = ModelWithAccess(name=model_name, max_tokens=tokens, has_access=access_granted)
            available_models.append(model_instance)

    return available_models
