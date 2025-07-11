from typing import Any

import httpx
from openai import AsyncOpenAI

from granite_guardian_shield.config import GraniteGuardianShieldConfig
from granite_guardian_shield.inference import GraniteGuardianVLLMInference
from granite_guardian_shield.shield import GraniteGuardianShield


async def get_adapter_impl(config: GraniteGuardianShieldConfig, _deps) -> Any:
    # Initialize OpenAI Client based on user's configuration
    api_key = config.api_key
    openai_client: AsyncOpenAI = AsyncOpenAI(
        base_url=config.base_url,
        api_key=api_key.get_secret_value() if api_key else None,
        http_client=httpx.AsyncClient(verify=config.verify_ssl),
    )
    inference = GraniteGuardianVLLMInference(openai_client, config.model)

    # Initialize the Granite Guardian shields manager
    impl = GraniteGuardianShield(inference)
    await impl.initialize()
    return impl
