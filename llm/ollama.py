import asyncio
from typing import AsyncIterator, List, Dict, Any
from PIL import Image
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionChunk
from abstract_llm import abc_llm

class ollama(abc_llm):
    def __init__(self,
                 model: str,
                 base_url: str,
                 api_key: str = 'z',
                 org_id: str = 'z',
                 proj_id: str = 'z',
                 temperature: float = 1.0,
                 ):

        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=org_id,
            project=proj_id,
            api_key=api_key,
        )

    async def text_chat(self,
                        messages: List[Dict[str, Any]],
                        system: str = None,
                        ) -> AsyncIterator[str]:
        stream = None
        try:
            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]

            stream: AsyncStream[
                ChatCompletionChunk
            ] = await self.client.chat.completions.create(
                messages=messages_with_system,
                model=self.model,
                stream=True,
                temperature=self.temperature,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                yield chunk.choices[0].delta.content

        except APIConnectionError as e:
            # logger.error(
            #     f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            # )
            # yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"
            yield  f"error: {e}"
        except RateLimitError as e:
            # logger.error(
            #     f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            # )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
            # logger.error(f"LLM API: Error occurred: {e}")
            # logger.info(f"Base URL: {self.base_url}")
            # logger.info(f"Model: {self.model}")
            # logger.info(f"Messages: {messages}")
            # logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                # logger.debug("Chat completion finished.")
                await stream.close()
                # logger.debug("Stream closed.")

    async def image_chat(self,
                         image: Image.Image | str | bytes,
                         system: str = None,
                         ) -> AsyncIterator[str]:
        pass

if __name__ == "__main__":
    # deepseek-r1:1.5b
    async def main():
        # 初始化 LLM 客户端（Ollama 通常不需要 API 密钥，用空字符串占位）
        llm_test = ollama(
            base_url="http://localhost:11434/v1",
            model="deepseek-r1:1.5b",
            api_key="",  # Ollama 不需要密钥，但参数为必填项
            temperature=0.7
        )

        messages = [{"role": "user", "content": "你好！"}]

        try:
            # 使用异步生成器获取流式响应
            async for chunk in llm_test.text_chat(
                    messages=messages,
                    system="你是一个乐于助人的助手"
            ):
                print(chunk, end="", flush=True)  # 实时输出
        except Exception as e:
            print(f"\n发生错误: {str(e)}")

    asyncio.run(main())
