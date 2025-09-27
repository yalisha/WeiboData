#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weibo Advanced Search Crawler (daily, keyword-based)

功能：
- 通过“高级搜索”URL参数进行按天检索，分页抓取搜索结果（默认最多 50 页/天）。
- 抽取文本、时间、作者、互动数、帖子链接与页面图片（缩略图），并保存至 CSV + SQLite。
- 可选代理（简单模式）。

使用前提：
- 需要提供已登录微博账号的 Cookie（从浏览器复制），填入同目录 config.yaml。
- 合理控制频率、遵守平台条款，仅用于研究用途。

默认日期范围：2022-01-01 至今天（可通过参数覆盖）。

示例：
  python weibo_advanced_search.py --keyword 金价 \
    --start 2022-01-01 --end 2025-12-31 --download-images \
    --out-dir ./output --image-dir ./images

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception:
        print("[WARN] 未安装 pyyaml，尝试使用内置简易解析（key: value 格式）")
        return _load_simple_kv_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_simple_kv_yaml(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip("\"')")
    return data


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def daterange(start_date: dt.date, end_date: dt.date) -> Iterable[dt.date]:
    cur = start_date
    one_day = dt.timedelta(days=1)
    while cur <= end_date:
        yield cur
        cur += one_day


def parse_int_from_text(text: str) -> int:
    m = re.search(r"(\d+)", text.replace(",", ""))
    return int(m.group(1)) if m else 0


def normalize_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http"):
        return url
    return "https://" + url.lstrip("/")


@dataclasses.dataclass
class Post:
    post_id: str
    keyword: str
    created_at: str
    author_name: str
    author_home: str
    verified: int
    text: str
    reposts: int
    comments: int
    likes: int
    post_url: str
    day: str
    crawl_ts: int
    image_urls: List[str]
    image_paths: List[str]


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
              post_id TEXT PRIMARY KEY,
              keyword TEXT,
              created_at TEXT,
              author_name TEXT,
              author_home TEXT,
              verified INTEGER,
              text TEXT,
              reposts INTEGER,
              comments INTEGER,
              likes INTEGER,
              post_url TEXT,
              day TEXT,
              crawl_ts INTEGER
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
              post_id TEXT,
              img_index INTEGER,
              img_url TEXT,
              img_path TEXT,
              PRIMARY KEY (post_id, img_index)
            )
            """
        )
        self.conn.commit()

    def upsert_post(self, p: Post):
        c = self.conn.cursor()
        c.execute(
            """
            INSERT INTO posts(post_id, keyword, created_at, author_name, author_home, verified, text, reposts, comments, likes, post_url, day, crawl_ts)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(post_id) DO UPDATE SET
                keyword=excluded.keyword,
                created_at=excluded.created_at,
                author_name=excluded.author_name,
                author_home=excluded.author_home,
                verified=excluded.verified,
                text=excluded.text,
                reposts=excluded.reposts,
                comments=excluded.comments,
                likes=excluded.likes,
                post_url=excluded.post_url,
                day=excluded.day,
                crawl_ts=excluded.crawl_ts
            """,
            (
                p.post_id,
                p.keyword,
                p.created_at,
                p.author_name,
                p.author_home,
                p.verified,
                p.text,
                p.reposts,
                p.comments,
                p.likes,
                p.post_url,
                p.day,
                p.crawl_ts,
            ),
        )
        c.execute("DELETE FROM images WHERE post_id=?", (p.post_id,))
        for i, (u, path) in enumerate(zip(p.image_urls, p.image_paths)):
            c.execute(
                "INSERT OR REPLACE INTO images(post_id, img_index, img_url, img_path) VALUES(?,?,?,?)",
                (p.post_id, i, u, path),
            )
        self.conn.commit()

    def has_day(self, keyword: str, day: str) -> bool:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(1) FROM posts WHERE keyword=? AND day=?", (keyword, day))
        n = c.fetchone()[0]
        return n > 0


def build_session(cfg: dict) -> requests.Session:
    s = requests.Session()
    headers = cfg.get("headers", {}) or {}
    # 基本头
    headers.setdefault("User-Agent", "Mozilla/5.0")
    headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8")
    headers.setdefault("Accept-Language", "zh-CN,zh;q=0.9")
    headers.setdefault("Connection", "keep-alive")
    s.headers.update(headers)
    cookie = (cfg.get("cookie") or "").strip()
    if cookie:
        s.headers.update({"Cookie": cookie})
    return s


def build_proxies(cfg: dict) -> Optional[Dict[str, str]]:
    pxy = cfg.get("proxy", {}) or {}
    if not pxy or not pxy.get("enable"):
        return None
    mode = pxy.get("mode", "simple")
    if mode == "simple":
        http = pxy.get("http")
        https = pxy.get("https") or http
        if not (http or https):
            return None
        proxies = {}
        if http:
            proxies["http"] = http
        if https:
            proxies["https"] = https
        return proxies
    # 预留和 MediaCrawlerPro 的对接入口
    return None


def build_timescope(day: dt.date) -> str:
    d = day.strftime("%Y-%m-%d")
    return f"custom:{d}-0:{d}-23"


def make_search_url(keyword: str, day: dt.date, page: int) -> str:
    from urllib.parse import quote

    q = quote(keyword)
    timescope = build_timescope(day)
    # 以“全部 + 全部”作为默认筛选；后续可追加 ori/haspic 等参数
    return (
        "https://s.weibo.com/weibo?q="
        + q
        + f"&typeall=1&suball=1&timescope={timescope}&Refer=g&page={page}"
    )


def extract_post_id(url: str) -> str:
    # 尝试从 weibo.com/{uid}/{mblogid} 提取最后一段作为ID
    try:
        path = re.sub(r"\?.*$", "", url)
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2:
            return parts[-1]
    except Exception:
        pass
    return ""


def parse_search_html(html: str) -> Tuple[List[dict], bool]:
    """返回列表[ {字段...} ], 以及是否检测到“无结果/结束”的标志。"""
    soup = BeautifulSoup(html, "lxml")

    # 结束信号：
    if soup.select_one(".card-no-result, .no-result"):
        return [], True

    items: List[dict] = []
    for wrap in soup.select("div.card-wrap"):
        # 排除热门/置顶等非 feed_list_item 的卡片
        if wrap.get("action-type") != "feed_list_item":
            # 某些页面该属性在子节点
            if not wrap.select_one('[action-type="feed_list_item"]'):
                continue

        # 作者与内容
        name_a = wrap.select_one("div.card > div.info > div > a.name") or wrap.select_one(
            "div.info > a.name"
        )
        author_name = (name_a.get_text(strip=True) if name_a else "")
        author_home = normalize_url(name_a.get("href")) if name_a else ""

        # 文本内容
        txt_p = wrap.select_one("p.txt")
        text = txt_p.get_text(" ", strip=True) if txt_p else ""

        # 时间与链接
        from_p = wrap.select_one("p.from")
        created_at = ""
        post_url = ""
        if from_p:
            time_a = from_p.find("a")
            if time_a:
                created_at = time_a.get("title") or time_a.get_text(strip=True)
                post_url = normalize_url(time_a.get("href") or "")

        # 互动数
        reposts = comments = likes = 0
        act = wrap.select_one("div.card-act")
        if act:
            lis = act.find_all("li")
            if len(lis) >= 3:
                reposts = parse_int_from_text(lis[0].get_text(strip=True))
                comments = parse_int_from_text(lis[1].get_text(strip=True))
                likes = parse_int_from_text(lis[2].get_text(strip=True))

        # 认证
        verified = 1 if wrap.select_one("i.icon-vip, i.ico-vip, i.W_icon") else 0

        # 图片（页面图）
        image_urls: List[str] = []
        for img in wrap.select("div.media img, ul.m-auto-list img, ul.m-thumb img"):
            src = img.get("src") or img.get("data-src") or ""
            if not src:
                continue
            src = normalize_url(src)
            if src not in image_urls:
                image_urls.append(src)

        post_id = extract_post_id(post_url)
        if not post_id and wrap.get("mid"):
            post_id = wrap.get("mid")

        if not post_id and text:
            # 退化的去重键
            post_id = f"{hash(text)}_{hash(post_url)}"

        items.append(
            dict(
                post_id=post_id,
                author_name=author_name,
                author_home=author_home,
                text=text,
                created_at=created_at,
                post_url=post_url,
                reposts=reposts,
                comments=comments,
                likes=likes,
                verified=verified,
                image_urls=image_urls,
            )
        )

    return items, False


def download_image(session: requests.Session, url: str, out: Path, proxies=None) -> Optional[Path]:
    try:
        r = session.get(url, timeout=20, proxies=proxies, headers={"Referer": "https://s.weibo.com/"})
        if r.status_code == 200 and r.content:
            ensure_dir(out.parent)
            out.write_bytes(r.content)
            return out
    except Exception as e:
        print(f"[WARN] 下载图片失败: {e} -> {url}")
    return None


def write_csv(day_csv: Path, posts: List[Post]):
    ensure_dir(day_csv.parent)
    fieldnames = [
        "post_id",
        "keyword",
        "created_at",
        "author_name",
        "author_home",
        "verified",
        "text",
        "reposts",
        "comments",
        "likes",
        "post_url",
        "day",
        "crawl_ts",
        "image_urls",
        "image_paths",
    ]
    is_new = not day_csv.exists()
    with open(day_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        for p in posts:
            w.writerow(
                {
                    "post_id": p.post_id,
                    "keyword": p.keyword,
                    "created_at": p.created_at,
                    "author_name": p.author_name,
                    "author_home": p.author_home,
                    "verified": p.verified,
                    "text": p.text,
                    "reposts": p.reposts,
                    "comments": p.comments,
                    "likes": p.likes,
                    "post_url": p.post_url,
                    "day": p.day,
                    "crawl_ts": p.crawl_ts,
                    "image_urls": ";".join(p.image_urls),
                    "image_paths": ";".join([str(x) for x in p.image_paths]),
                }
            )


def crawl_day(
    session: requests.Session,
    store: SQLiteStore,
    keyword: str,
    day: dt.date,
    out_dir: Path,
    img_dir: Path,
    max_pages: int,
    download_images: bool,
    min_interval: float,
    max_interval: float,
    proxies: Optional[Dict[str, str]] = None,
) -> Tuple[int, int]:
    page = 1
    seen_ids = set()
    collected: List[Post] = []
    day_str = day.strftime("%Y-%m-%d")
    while page <= max_pages:
        url = make_search_url(keyword, day, page)
        print(f"[INFO] {day_str} 第{page}页: {url}")
        resp = session.get(url, timeout=30, proxies=proxies)
        if resp.status_code != 200:
            print(f"[WARN] 状态码 {resp.status_code}, 停止该日")
            break
        items, is_end = parse_search_html(resp.text)
        if is_end or not items:
            print("[INFO] 无结果或到达末尾，结束该日分页。")
            break

        page_first_id = items[0].get("post_id")
        if page_first_id and page_first_id in seen_ids:
            print("[INFO] 检测到重复内容，提前停止该日分页。")
            break

        for it in items:
            pid = it.get("post_id") or ""
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            created_at = it.get("created_at") or ""
            crawl_ts = int(time.time())
            img_urls: List[str] = it.get("image_urls", [])
            img_paths: List[str] = []

            # 下载图片（页面图缩略图）
            if download_images and img_urls:
                for idx, u in enumerate(img_urls):
                    # 文件名：postid_idx.jpg（不保证原格式，统一 .jpg）
                    fname = f"{pid or hash(it.get('text', ''))}_{idx}.jpg"
                    out_path = img_dir / keyword / day_str / fname
                    saved = download_image(session, u, out_path, proxies=proxies)
                    img_paths.append(str(saved) if saved else "")

            p = Post(
                post_id=pid,
                keyword=keyword,
                created_at=created_at,
                author_name=it.get("author_name", ""),
                author_home=it.get("author_home", ""),
                verified=int(it.get("verified", 0)),
                text=it.get("text", ""),
                reposts=int(it.get("reposts", 0)),
                comments=int(it.get("comments", 0)),
                likes=int(it.get("likes", 0)),
                post_url=it.get("post_url", ""),
                day=day_str,
                crawl_ts=crawl_ts,
                image_urls=img_urls,
                image_paths=img_paths,
            )
            collected.append(p)
            store.upsert_post(p)

        # 持久化每日 CSV 追加
        csv_path = out_dir / keyword / f"{day_str}.csv"
        write_csv(csv_path, collected)
        collected.clear()

        # 间隔
        time.sleep(random.uniform(min_interval, max_interval))
        page += 1

    return len(seen_ids), page - 1


def parse_args(cfg: dict) -> argparse.Namespace:
    defaults = cfg.get("defaults", {}) or {}
    parser = argparse.ArgumentParser(description="Weibo 高级搜索（日粒度）爬虫")
    parser.add_argument("--keyword", required=True, help="关键词（单个）")
    parser.add_argument("--start", default=defaults.get("start_date", "2022-01-01"))
    parser.add_argument("--end", default=defaults.get("end_date", ""))
    parser.add_argument("--max-pages-per-day", type=int, default=int(defaults.get("max_pages_per_day", 50)))
    parser.add_argument("--download-images", action="store_true", default=True)
    parser.add_argument("--no-download-images", action="store_false", dest="download_images")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "output"))
    parser.add_argument("--image-dir", default=str(BASE_DIR / "images"))
    parser.add_argument("--db", default=str(BASE_DIR / "weibo_search.db"))
    parser.add_argument("--min-interval", type=float, default=float(defaults.get("min_interval", 1.2)))
    parser.add_argument("--max-interval", type=float, default=float(defaults.get("max_interval", 3.5)))
    parser.add_argument("--proxy-enable", action="store_true", default=bool(cfg.get("proxy", {}).get("enable", False)))
    parser.add_argument("--resume", action="store_true", default=True, help="跳过已抓取过的日期（依据DB+CSV存在）")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    # 配置文件路径：若指定则优先该路径；否则采用默认搜索顺序
    parser.add_argument("--config", default=None, help="配置文件路径（可选）")
    return parser


def locate_config(cli_config: Optional[str]) -> Tuple[Optional[Path], dict]:
    candidates: List[Path] = []
    if cli_config:
        candidates.append(Path(cli_config).expanduser())
    # 1) 当前工作目录
    candidates.append(Path.cwd() / "config.yaml")
    # 2) 脚本同目录
    candidates.append(BASE_DIR / "config.yaml")

    for p in candidates:
        if p.exists():
            try:
                cfg = load_yaml(p)
                return p, cfg
            except Exception:
                pass
    return None, {}


def main():
    # 先只解析 --config 选项
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", default=None)
    known, _ = temp_parser.parse_known_args()

    cfg_path, cfg = locate_config(known.config)
    parser = parse_args(cfg)
    args = parser.parse_args()

    # 解析日期
    try:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    except Exception:
        start = dt.date(2022, 1, 1)
    if args.end:
        try:
            end = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
        except Exception:
            end = dt.date.today()
    else:
        end = dt.date.today()

    # 限制到 2025 年底（按你的默认约束，可根据需要放开）
    if end.year > 2025:
        end = dt.date(2025, 12, 31)

    # 若 CLI 再次提供 --config，以 CLI 为准并重新加载
    if args.config and (not cfg_path or Path(args.config).expanduser() != cfg_path):
        cfg_path, cfg = locate_config(args.config)

    if cfg_path:
        print(f"[INFO] 使用配置文件: {cfg_path}")
    else:
        print("[WARN] 未找到配置文件，将使用内置默认参数与空Cookie。")

    session = build_session(cfg)
    proxies = build_proxies(cfg) if args.proxy_enable else None

    store = SQLiteStore(Path(args.db))
    out_dir = Path(args.out_dir)
    img_dir = Path(args.image_dir)

    keyword = args.keyword
    max_pages = args.max_pages_per_day

    total_posts = 0
    total_days = 0
    for day in daterange(start, end):
        day_str = day.strftime("%Y-%m-%d")
        # 断点续爬：若 DB 中已有该日对应关键词的帖子且 CSV 文件存在，则跳过
        csv_path = Path(args.out_dir) / keyword / f"{day_str}.csv"
        if args.resume and store.has_day(keyword, day_str) and csv_path.exists():
            print(f"[SKIP] {day_str} 已存在数据（开启 --resume），跳过。")
            continue

        print(f"\n[DAY] {day_str} 开始抓取……")
        try:
            n_posts, n_pages = crawl_day(
                session,
                store,
                keyword,
                day,
                out_dir,
                img_dir,
                max_pages,
                args.download_images,
                args.min_interval,
                args.max_interval,
                proxies=proxies,
            )
            print(f"[DAY] {day_str} 完成：{n_posts}条，{n_pages}页")
            if n_posts > 0:
                total_days += 1
            total_posts += n_posts
        except KeyboardInterrupt:
            print("[INFO] 用户中断。")
            break
        except Exception as e:
            print(f"[ERROR] {day_str} 失败：{e}")

    print(f"\n[SUMMARY] 完成：{total_posts} 条帖子，覆盖 {total_days} 个有结果的日期。")


if __name__ == "__main__":
    main()
