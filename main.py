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
                    if seg.url and (img := await self._load_bytes(s_chain.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(s_chain.file)):
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
        
        # 配置验证
        api_url = self.conf.get("api_url")
        model = self.conf.get("model", "nano-banana")
        api_keys = self.conf.get("api_keys", [])
        
        logger.info("=== FigurinePro 插件配置检查 ===")
        logger.info(f"API URL: {api_url}")
        logger.info(f"Model: {model}")
        logger.info(f"API Keys 数量: {len(api_keys)}")
        
        if not api_url:
            logger.error("❌ 未配置 api_url")
        if not api_keys:
            logger.warning("⚠️ 未配置 API 密钥")
            
        logger.info("FigurinePro 插件已加载 (支持 Gemini SDK 和 OpenAI 格式)")

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

    # 添加缺失的数据加载方法
    async def _load_user_counts(self):
        """加载用户次数数据"""
        if not self.user_counts_file.exists(): 
            self.user_counts = {}
            return
            
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): 
                self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True)
            self.user_counts = {}

    async def _load_group_counts(self):
        """加载群组次数数据"""
        if not self.group_counts_file.exists(): 
            self.group_counts = {}
            return
            
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.group_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): 
                self.group_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True)
            self.group_counts = {}

    async def _load_user_checkin_data(self):
        """加载用户签到数据"""
        if not self.user_checkin_file.exists(): 
            self.user_checkin_data = {}
            return
            
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_checkin_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): 
                self.user_checkin_data = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户签到文件时发生错误: {e}", exc_info=True)
            self.user_checkin_data = {}

    # 这里继续原有的其他方法...
    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent):
        # ... 保持原有的 on_figurine_request 方法代码
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
        text = event.message_str.strip()
        if not text: return
        cmd = text.split()[0].strip()
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False
        if cmd == bnn_command:
            user_prompt = text.removeprefix(cmd).strip()
            is_bnn = True
            if not user_prompt: return
        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)
        else:
            return
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。");
                    return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。");
                return
        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。");
                return
        images_to_process = []
        display_cmd = cmd
        if is_bnn:
            MAX_IMAGES = 5
            original_count = len(img_bytes_list)
            if original_count > MAX_IMAGES:
                images_to_process = img_bytes_list[:MAX_IMAGES]
                yield event.plain_result(f"🎨 检测到 {original_count} 张图片，已选取前 {MAX_IMAGES} 张…")
            else:
                images_to_process = img_bytes_list
            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
            yield event.plain_result(f"🎨 检测到 {len(images_to_process)} 张图片，正在生成 [{display_cmd}]...")
        else:
            if not img_bytes_list:
                yield event.plain_result("请发送或引用一张图片。");
                return
            images_to_process = [img_bytes_list[0]]
            yield event.plain_result(f"🎨 收到请求，正在生成 [{cmd}]...")
        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {display_cmd}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(
                    f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(
                    f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    # 继续添加其他必要的方法...
    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0: 
            self.user_counts[user_id_str] = count - 1
            await self._save_user_counts()

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None,
                                                   functools.partial(json.dumps, self.user_counts, ensure_ascii=False,
                                                                     indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        count = self._get_group_count(group_id_str)
        if count > 0: 
            self.group_counts[group_id_str] = count - 1
            await self._save_group_counts()

    async def _save_group_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None,
                                                   functools.partial(json.dumps, self.group_counts, ensure_ascii=False,
                                                                     indent=4))
            await loop.run_in_executor(None, self.group_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    async def _save_user_checkin_data(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_checkin_data,
                                                                           ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_checkin_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户签到文件时发生错误: {e}", exc_info=True)

    # 这里继续添加其他原有方法...
    # 包括：on_text_to_image_request, add_lm_prompt, on_prompt_help, is_global_admin, 
    # on_checkin, on_add_user_counts, on_add_group_counts, on_query_counts, 
    # on_add_key, on_list_keys, on_delete_key, _get_api_key, _extract_image_url_from_response,
    # _call_gemini_api, _call_openai_api, _call_api, _is_gemini_api, terminate 等方法

    # 为了简洁，这里只展示关键方法，你需要确保所有原有方法都存在

    async def terminate(self):
        if self.iwf: 
            await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")
