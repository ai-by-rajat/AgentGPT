from typing import Any, Dict, Optional, Tuple, Type, Union

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from pydantic import Field

from reworkd_platform.schemas.agent import LLM_Model, ModelSettings
from reworkd_platform.schemas.user import UserBase
from reworkd_platform.settings import Settings


class WrappedChatOpenAI(ChatOpenAI):
    client: Any = Field(
        default=None,
        description="Meta private value but mypy will complain its missing",
    )
    max_tokens: int
    model_name: LLM_Model = Field(alias="model")


class WrappedAzureChatOpenAI(AzureChatOpenAI, WrappedChatOpenAI):
    openai_api_base: str
    openai_api_version: str
    deployment_name: str


WrappedChat = Union[WrappedAzureChatOpenAI, WrappedChatOpenAI]


def create_model(
    settings: Settings,
    model_settings: ModelSettings,
    user: UserBase,
    streaming: bool = False,
    force_model: Optional[LLM_Model] = None,
) -> WrappedChat:
    llm_model = force_model or model_settings.model
    model: Type[WrappedChat] = WrappedChatOpenAI
    base, headers, use_helicone, provider = get_base_and_headers(
        settings, model_settings, user
    )

    # Determine the actual model name to be passed to the LLM
    actual_llm_model_name = llm_model
    if provider == "ollama" and isinstance(llm_model, str) and llm_model.startswith("ollama/"):
        actual_llm_model_name = llm_model.split("/", 1)[1]

    # API key handling
    if provider == "ollama":
        # Ollama doesn't use an API key in the traditional sense via ChatOpenAI,
        # but ChatOpenAI class requires a non-empty openai_api_key.
        api_key_to_use = "ollama"
    elif model_settings.custom_api_key:
        api_key_to_use = model_settings.custom_api_key
    elif settings.openai_api_key and settings.openai_api_key != "<Should be updated via env>":
        api_key_to_use = settings.openai_api_key
    else:
        # This case should ideally not be hit if configuration is correct
        # and an appropriate provider (Ollama/Custom/Azure/OpenAI with key) is chosen.
        # If it is, it means we are trying to use a provider (likely OpenAI) without a valid key.
        # Forcing a dummy key here for ChatOpenAI if no other key is found and provider isn't ollama.
        # This might still lead to auth errors with actual OpenAI, but satisfies ChatOpenAI's requirement.
        api_key_to_use = "none"


    kwargs = {
        "openai_api_base": base,
        "openai_api_key": api_key_to_use,
        "temperature": model_settings.temperature,
        "model": actual_llm_model_name,
        "max_tokens": model_settings.max_tokens,
        "streaming": streaming,
        "max_retries": 5,
        "model_kwargs": {"user": user.email, "headers": headers},
    }

    if provider == "azure":
        model = WrappedAzureChatOpenAI
        deployment_name = llm_model.replace(".", "")
        kwargs.update(
            {
                "openai_api_version": settings.openai_api_version,
                "deployment_name": deployment_name,
                "openai_api_type": "azure",
                "openai_api_base": base.rstrip("v1") if base else None,
            }
        )
        if use_helicone: # Helicone with Azure not fully supported yet by Helicone
            kwargs["model"] = deployment_name


    return model(**kwargs)  # type: ignore


def get_base_and_headers(
    settings_: Settings, model_settings: ModelSettings, user: UserBase
) -> Tuple[Optional[str], Optional[Dict[str, str]], bool, Optional[str]]:
    use_custom_key = bool(model_settings.custom_api_key)
    use_ollama = settings_.ollama_enabled and not use_custom_key
    use_azure = (
        settings_.openai_api_base
        and "azure" in settings_.openai_api_base
        and not use_custom_key
        and not use_ollama
    )
    use_helicone = settings_.helicone_enabled and not use_custom_key and not use_ollama

    provider: Optional[str] = None
    base: Optional[str] = None
    headers: Optional[Dict[str, str]] = None

    if use_ollama:
        provider = "ollama"
        base = settings_.ollama_api_base
    elif use_custom_key:
        provider = "custom_openai"
        base = "https://api.openai.com/v1"  # Default for custom keys, can be overridden by model_settings if needed
    elif use_azure:
        provider = "azure"
        base = settings_.openai_api_base
    elif use_helicone:
        provider = "helicone"
        base = settings_.helicone_api_base
        headers = {
            "Helicone-Auth": f"Bearer {settings_.helicone_api_key}",
            "Helicone-Cache-Enabled": "true",
            "Helicone-User-Id": user.id,
            "Helicone-OpenAI-Api-Base": settings_.openai_api_base or "https://api.openai.com/v1",
        }
    elif settings_.openai_api_base:
        provider = "openai"
        base = settings_.openai_api_base
    elif not settings_.ollama_enabled and not use_custom_key and not use_azure and not use_helicone:
        # This case means no specific provider is configured other than potentially plain OpenAI.
        # Default to public OpenAI API if an API key is available, otherwise no provider can be determined.
        if settings_.openai_api_key and settings_.openai_api_key != "<Should be updated via env>":
            provider = "openai"
            base = "https://api.openai.com/v1"
        else:
            # No valid provider could be determined with the current settings
            provider = None
            base = None
    # If ollama_enabled is true, and other conditions didn't match (e.g. custom key),
    # it should have been caught by the `if use_ollama:` block.
    # If we reach here and provider is still None, it means no valid configuration was found.

    return base, headers, use_helicone, provider
