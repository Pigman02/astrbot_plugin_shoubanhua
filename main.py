import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage
from google import genai
from google.genai.types import HttpOptions
from io import BytesIO

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw", 
    "通过第三方api进行手办化等功能",
    "1.4.0",
    "https://github.com/shkjw/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url: logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在尝试下载图片: {url}")
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()
                    return await resp.read()
            except aiohttp.ClientResponseError as e:
                logger.error(f"图片下载失败: HTTP状态码 {e.status}, URL: {url}, 原因: {e.message}")
                return None
            except asyncio.TimeoutError:
                logger.error(f"图片下载失败: 请求超时 (30s), URL: {url}")
                return None
            except Exception as e:
                logger.error(f"图片下载失败: 发生未知错误, URL: {url}, 错误类型: {type(e).__name__}, 错误: {e}",
                             exc_info=True)
                return None

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit(): logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。"); return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e:
                logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw: return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            if avatar := await self._get_avatar(event.get_sender_id()):
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}
        self.key_index = 0
        self.key_lock = asyncio.Lock()
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        logger.info("FigurinePro 插件已加载 (支持 Gemini SDK 和 OpenAI 格式)")
        if not self.conf.get("api_keys"):
            logger.warning("FigurinePro: 未配置任何 API 密钥，插件可能无法工作")

    async def _load_prompt_map(self):
        logger.info("正在加载 prompts...")
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            try:
                if ":" in item:
                    key, value = item.split(":", 1)
                    self.prompt_map[key.strip()] = value.strip()
                else:
                    logger.warning(f"跳过格式错误的 prompt (缺少冒号): {item}")
            except ValueError:
                logger.warning(f"跳过格式错误的 prompt: {item}")
        logger.info(f"加载了 {len(self.prompt_map)} 个 prompts。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        # ... 保持不变 ...
        # 原有的 on_figurine_request 方法代码保持不变
        pass

    @filter.command("文生图", prefix_optional=True)
    async def on_text_to_image_request(self, event: AstrMessageEvent):
        # ... 保持不变 ...
        # 原有的 on_text_to_image_request 方法代码保持不变
        pass

    # ... 其他方法保持不变 ...

    async def _get_api_key(self) -> str | None:
        keys = self.conf.get("api_keys", [])
        if not keys: return None
        async with self.key_lock:
            key = keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(keys)
            return key

    def _is_gemini_api(self, api_url: str) -> bool:
        """判断是否为 Gemini API"""
        gemini_domains = ['generativelanguage.googleapis.com', 'googleapis.com']
        return any(domain in api_url for domain in gemini_domains)

    async def _call_gemini_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        """使用 Gemini SDK 调用 API"""
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"
        
        try:
            # 配置 HTTP 选项
            http_options = HttpOptions(
                base_url=self.conf.get("api_url", "https://generativelanguage.googleapis.com")
            )
            
            # 创建 Gemini 客户端
            client = genai.Client(api_key=api_key, http_options=http_options)
            
            # 构建内容
            contents = []
            if prompt:
                contents.append(prompt)
            
            # 添加图片
            for image_bytes in image_bytes_list:
                pil_image = PILImage.open(BytesIO(image_bytes))
                contents.append(pil_image)
            
            if not contents:
                return "没有有效的内容发送给 Gemini API"

            # 调用 API
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="models/" + self.conf.get("model", "gemini-2.0-flash-exp"),
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )

            # 处理响应
            if not response or not hasattr(response, 'candidates') or not response.candidates:
                return "Gemini API 返回空响应"

            candidate = response.candidates[0]
            
            # 检查安全限制
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == 'SAFETY':
                return "内容因安全策略被阻止"

            if not (hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts')):
                return "Gemini API 返回内容格式错误"

            # 提取生成的图片
            for part in candidate.content.parts:
                if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.mime_type.startswith('image/'):
                    img_data = part.inline_data.data
                    return img_data  # 直接返回图片字节数据

            return "Gemini API 未生成图片"

        except Exception as e:
            logger.error(f"Gemini API 调用失败: {e}", exc_info=True)
            return f"Gemini API 调用失败: {str(e)}"

    async def _call_openai_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        """使用 OpenAI 格式调用 API"""
        api_url = self.conf.get("api_url")
        if not api_url: return "API URL 未配置"
        
        api_key = await self._get_api_key()
        if not api_key: return "无可用的 API Key"
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        # 构建 content 列表
        content = [{"type": "text", "text": prompt}]
        for image_bytes in image_bytes_list:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        # 构建请求载荷
        payload = {
            "model": self.conf.get("model", "nano-banana"),
            "max_tokens": 1500,
            "stream": False,
            "messages": [{"role": "user", "content": content}]
        }

        try:
            if not self.iwf: return "ImageWorkflow 未初始化"
            
            async with self.iwf.session.post(api_url, json=payload, headers=headers, proxy=self.iwf.proxy,
                                             timeout=120) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"API 请求失败: HTTP {resp.status}, 响应: {error_text}")
                    return f"API请求失败 (HTTP {resp.status}): {error_text[:200]}"
                
                data = await resp.json()
                if "error" in data: 
                    return data["error"].get("message", json.dumps(data["error"]))
                
                # 提取图片 URL
                gen_image_url = self._extract_image_url_from_response(data)
                if not gen_image_url:
                    error_msg = f"API响应中未找到图片数据: {str(data)[:500]}..."
                    logger.error(f"API响应中未找到图片数据: {data}")
                    return error_msg
                
                if gen_image_url.startswith("data:image/"):
                    b64_data = gen_image_url.split(",", 1)[1]
                    return base64.b64decode(b64_data)
                else:
                    return await self.iwf._download_image(gen_image_url) or "下载生成的图片失败"
                    
        except asyncio.TimeoutError:
            logger.error("API 请求超时")
            return "请求超时"
        except Exception as e:
            logger.error(f"调用 API 时发生未知错误: {e}", exc_info=True)
            return f"发生未知错误: {e}"

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        """统一的 API 调用入口，自动选择调用方式"""
        api_url = self.conf.get("api_url", "")
        
        # 根据 API URL 自动选择调用方式
        if self._is_gemini_api(api_url):
            logger.info("使用 Gemini SDK 调用 API")
            return await self._call_gemini_api(image_bytes_list, prompt)
        else:
            logger.info("使用 OpenAI 格式调用 API")
            return await self._call_openai_api(image_bytes_list, prompt)

    def _extract_image_url_from_response(self, data: Dict[str, Any]) -> str | None:
        """从 OpenAI 格式响应中提取图片 URL（保持不变）"""
        try:
            return data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        except (IndexError, TypeError, KeyError):
            pass
        try:
            return data["choices"][0]["message"]["images"][0]["url"]
        except (IndexError, TypeError, KeyError):
            pass
        try:
            content_text = data["choices"][0]["message"]["content"]
            url_match = re.search(r'https?://[^\s<>")\]]+', content_text)
            if url_match: return url_match.group(0).rstrip(")>,'\"")
            if '![image](' in content_text:
                start_idx = content_text.find('![image](') + len('![image](')
                end_idx = content_text.find(')', start_idx)
                if end_idx > start_idx:
                    return content_text[start_idx:end_idx].strip()
        except (IndexError, TypeError, KeyError):
            pass
        return None

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
