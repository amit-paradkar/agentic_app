import logging
from typing import Union
from services.grok_service import GroqService
from services.elevenlabs_service import ElevenLabsService


class ChatUseCases:
    """A class to handle use cases of AIDost Application."""

    def __init__(self):
        """Initialize the ImageToText class and validate environment variables."""
        self._groq_service = GroqService()
        self._elevenlabs_service = ElevenLabsService()
        self.logger = logging.getLogger(__name__)

    #def process_user_query(query_mode: str, query_data: Union[str, bytes]):
    def convert_user_input(self, from_mode: str, to_mode: str, data: Union[str, bytes]):
        """
        """
        content = ""
        
        if from_mode == "image" and to_mode == "text":
            # Analyze image and add to message content
            try:
                # Use global ImageToText instance
                description = await self.groq_service.analyze_image(
                    data,
                    "Please describe what you see in this image in the context of our conversation.",
                )
                content += f"\n[Image Analysis: {description}]"
            except Exception as e:
                self.logger.warning(f"Failed to analyze image: {e}")
                return ""
        elif from_mode == "text" and to_mode == "image":
            return ""
        elif from_mode == "voice" and to_mode == "text":
            content = await self.elevenlabs_service.transcribe(data)
            return content
        elif from_mode == "text" and to_mode == "voice":
            content = await self.elevenlabs_service.synthesize(data)
            return content
        
        return content