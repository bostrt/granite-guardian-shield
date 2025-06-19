from typing import Any

from granite_guardian_shield.config import GraniteGuardianShieldConfig


async def get_adapter_impl(config: GraniteGuardianShieldConfig, _deps) -> Any:
    from granite_guardian_shield.shield import GraniteGuardianShield

    impl = GraniteGuardianShield(config)
    await impl.initialize()
    return impl