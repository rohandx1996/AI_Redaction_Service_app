import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Environment settings required by the integrated Gateway client."""

    @property
    def bosch_gateway_url(self) -> str:
        return os.getenv("BOSCH_GATEWAY_URL", "")

    @property
    def bosch_gateway_token(self) -> str:
        return os.getenv("BOSCH_GATEWAY_TOKEN", "")


settings = Settings()
