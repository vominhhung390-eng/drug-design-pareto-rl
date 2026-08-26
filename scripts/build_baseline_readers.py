from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lxml import etree
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE = Path(r"D:\自身方法改造\docs\基线论文")
SOURCE = BASE / "_source"
OUTPUT = BASE / "中文译读"


@dataclass(frozen=True)
class Paper:
    order: int
    key: str
    pmcid: str
    xml_name: str
    zip_name: str
    pdf_name: str
    doi: str
    license: str
    mode: str


PAPERS = [
    Paper(1, "POLYGON", "PMC11074339", "POLYGON_PMC11074339.xml", "POLYGON_PMC11074339_SupplementaryFiles.zip", "01_POLYGON_De_novo_generation_of_multi-target_compounds.pdf", "10.1038/s41467-024-47120-y", "CC BY 4.0", "full"),
    Paper(2, "REINVENT4", "PMC10882833", "REINVENT4_PMC10882833.xml", "REINVENT4_PMC10882833_SupplementaryFiles.zip", "02_REINVENT4_Modern_AI-driven_generative_molecule_design.pdf", "10.1186/s13321-024-00812-5", "CC BY 4.0", "full"),
    Paper(3, "DrugEx_v2", "PMC8588612", "DrugEx_v2_PMC8588612.xml", "DrugEx_v2_PMC8588612_SupplementaryFiles.zip", "03_DrugEx_v2_Pareto_multi-objective_RL.pdf", "10.1186/s13321-021-00561-9", "CC BY 4.0", "full"),
    Paper(4, "MO-LSO", "PMC11573897", "MO-LSO_PMC11573897.xml", "MO-LSO_PMC11573897_SupplementaryFiles.zip", "04_MO-LSO_Multi-objective_latent_space_optimization.pdf", "10.1016/j.patter.2024.101042", "CC BY-NC-ND 4.0", "interpretation"),
    Paper(5, "Mothra", "PMC11481094", "Mothra_PMC11481094.xml", "Mothra_PMC11481094_SupplementaryFiles.zip", "05_Mothra_Multiobjective_de_novo_MCTS.pdf", "10.1021/acs.jcim.4c00759", "CC BY 4.0", "full"),
]


TERMS = {
    "reinforcement learning": "强化学习",
    "multi-objective": "多目标",
    "Pareto front": "帕累托前沿",
    "Pareto efficiency": "帕累托效率",
    "latent space": "潜在空间",
    "de novo": "从头生成",
    "drug-likeness": "类药性",
    "synthetic accessibility": "合成可及性",
    "Monte Carlo tree search": "蒙特卡洛树搜索",
    "recurrent neural network": "循环神经网络",
    "variational autoencoder": "变分自编码器",
    "transfer learning": "迁移学习",
    "curriculum learning": "课程学习",
    "polypharmacology": "多靶点药理学",
}

POST_TRANSLATION_FIXES = {
    "重塑 4": "REINVENT 4",
    "重塑4": "REINVENT 4",
    "重塑 2.0": "REINVENT 2.0",
    "重塑2.0": "REINVENT 2.0",
    "重塑": "REINVENT",
    "魔斯拉": "Mothra",
    "莫斯拉": "Mothra",
    "化学生物学BL": "ChEMBL",
    "Pareto 前端": "帕累托前沿",
    "Pareto前端": "帕累托前沿",
    "超交易量": "超体积",
    "双硼酸正切": "双曲正切",
    "分子发生器": "分子生成器",
}


def clean_text(node) -> str:
    text = " ".join("".join(node.itertext()).split())
    return re.sub(r"\s+([,.;:!?%])", r"\1", text).strip()


def local_name(node) -> str:
    return etree.QName(node).localname


def xpath_of(node) -> str:
    return node.getroottree().getpath(node)


def split_for_translation(text: str, limit: int = 1800) -> list[str]:
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(sentence) > limit:
            words = sentence.split()
            for word in words:
                if current and len(current) + len(word) + 1 > limit:
                    chunks.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
        elif current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def google_translate_once(text: str) -> str:
    url = (
        "https://translate.googleapis.com/translate_a/single?client=gtx"
        "&sl=en&tl=zh-CN&dt=t&q=" + urllib.parse.quote(text)
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.load(response)
    translated = "".join(part[0] for part in data[0] if part and part[0])
    return translated.strip()


def translate_uncached(text: str) -> str:
    if not text.strip():
        return ""
    chunks = split_for_translation(text)
    result = []
    for chunk in chunks:
        error = None
        for attempt in range(5):
            try:
                result.append(google_translate_once(chunk))
                error = None
                break
            except Exception as exc:  # network retry
                error = exc
                time.sleep(1.5 * (attempt + 1))
        if error is not None:
            result.append("【机器翻译暂缺】" + chunk)
    translated = " ".join(result)
    for en, zh in TERMS.items():
        translated = re.sub(re.escape(en), zh, translated, flags=re.I)
    return translated


class Translator:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self.cache = {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def translate_many(self, texts: list[str], workers: int = 6) -> None:
        unique = {self.key(text): text for text in texts if text.strip()}
        pending = [(key, text) for key, text in unique.items() if key not in self.cache]
        if not pending:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(translate_uncached, text): (key, text) for key, text in pending}
            completed = 0
            for future in as_completed(futures):
                key, text = futures[future]
                try:
                    self.cache[key] = future.result()
                except Exception:
                    self.cache[key] = "【机器翻译暂缺】" + text
                completed += 1
                if completed % 25 == 0:
                    self.save()
                    print(f"  translated {completed}/{len(pending)}", flush=True)
        self.save()

    def get(self, text: str) -> str:
        translated = self.cache.get(self.key(text), "【机器翻译暂缺】" + text)
        for wrong, correct in POST_TRANSLATION_FIXES.items():
            translated = translated.replace(wrong, correct)
        return translated

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")


def metadata(root) -> dict:
    title_nodes = root.xpath("//article-meta/title-group/article-title")
    title = clean_text(title_nodes[0]) if title_nodes else "Untitled"
    journal_nodes = root.xpath("//journal-meta/journal-title-group/journal-title")
    journal = clean_text(journal_nodes[0]) if journal_nodes else ""
    authors = []
    for contrib in root.xpath("//article-meta/contrib-group/contrib[@contrib-type='author']"):
        surname = contrib.xpath(".//surname")
        given = contrib.xpath(".//given-names")
        name = " ".join([clean_text(given[0]) if given else "", clean_text(surname[0]) if surname else ""]).strip()
        if name:
            authors.append(name)
    year_nodes = root.xpath("//article-meta/pub-date/year")
    return {"title": title, "journal": journal, "authors": authors, "year": clean_text(year_nodes[0]) if year_nodes else ""}


def sentence_summary(text: str, max_sentences: int = 3) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 35]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text.lower())
    stop = {"the", "and", "for", "with", "that", "this", "from", "were", "was", "are", "into", "using", "their", "which", "have", "has", "been", "our", "we", "in", "of", "to", "a", "an", "is", "as", "by", "on", "or"}
    freq = Counter(w for w in words if w not in stop)
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = re.findall(r"[A-Za-z][A-Za-z-]{2,}", sentence.lower())
        score = sum(freq[t] for t in tokens if t in freq) / math.sqrt(max(len(tokens), 1))
        scored.append((score, index, sentence))
    chosen = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(item[2] for item in chosen)


def unpack_assets(paper: Paper, out_dir: Path) -> Path:
    extracted = out_dir / "_extracted"
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(SOURCE / paper.zip_name) as archive:
            for info in archive.infolist():
                target = (extracted / info.filename).resolve()
                if not str(target).startswith(str(extracted.resolve())):
                    continue
                archive.extract(info, extracted)
    return extracted


def find_graphic(extracted: Path, href: str) -> Path | None:
    if not href:
        return None
    href_name = Path(href).name.lower()
    candidates = [p for p in extracted.rglob("*") if p.is_file() and p.name.lower() == href_name]
    if candidates:
        return candidates[0]
    stem = Path(href_name).stem
    candidates = [p for p in extracted.rglob("*") if p.is_file() and p.stem.lower() == stem]
    return candidates[0] if candidates else None


def collect_blocks(root, mode: str) -> list[dict]:
    blocks: list[dict] = []
    counter = 0

    def add(kind: str, source: str, node, level: int = 2, extra: dict | None = None):
        nonlocal counter
        source = source.strip()
        if not source:
            return
        counter += 1
        block = {
            "id": f"b{counter:04d}",
            "type": kind,
            "source": source,
            "xpath": xpath_of(node),
            "level": level,
        }
        if extra:
            block.update(extra)
        blocks.append(block)

    abstracts = root.xpath("//article-meta/abstract")
    if abstracts:
        abstract_text = " ".join(clean_text(p) for p in abstracts[0].xpath(".//p"))
        if mode == "interpretation":
            abstract_text = sentence_summary(abstract_text, 5)
        add("section", "Abstract", abstracts[0], 2)
        add("paragraph", abstract_text, abstracts[0], 2)

    def visit(container, level: int):
        for child in container:
            name = local_name(child)
            if name == "title":
                add("section", clean_text(child), child, min(level, 5))
            elif name == "p":
                text = clean_text(child)
                if mode == "interpretation":
                    continue
                add("paragraph", text, child, level)
            elif name == "fig":
                label_nodes = child.xpath("./label")
                caption_nodes = child.xpath("./caption")
                label = clean_text(label_nodes[0]) if label_nodes else "Figure"
                caption = clean_text(caption_nodes[0]) if caption_nodes else ""
                graphics = child.xpath(".//graphic")
                href = ""
                if graphics:
                    href = graphics[0].get("{http://www.w3.org/1999/xlink}href", "")
                add("figure", f"{label}. {caption}".strip(), child, level, {"href": href})
            elif name == "table-wrap":
                label_nodes = child.xpath("./label")
                caption_nodes = child.xpath("./caption")
                label = clean_text(label_nodes[0]) if label_nodes else "Table"
                caption = clean_text(caption_nodes[0]) if caption_nodes else ""
                rows = []
                for tr in child.xpath(".//tr"):
                    cells = [clean_text(cell) for cell in tr.xpath("./th|./td")]
                    if cells:
                        rows.append(" | ".join(cells))
                table_text = f"{label}. {caption}\n" + "\n".join(rows)
                add("table", table_text.strip(), child, level)
            elif name == "list":
                items = [clean_text(item) for item in child.xpath("./list-item")]
                if mode == "full":
                    add("list", "\n".join(items), child, level)
            elif name == "disp-formula":
                add("formula", clean_text(child), child, level)
            elif name == "sec":
                if mode == "interpretation":
                    title_nodes = child.xpath("./title")
                    title = clean_text(title_nodes[0]) if title_nodes else "Section"
                    add("section", title, title_nodes[0] if title_nodes else child, min(level, 5))
                    paragraph_text = " ".join(clean_text(p) for p in child.xpath("./p"))
                    if paragraph_text:
                        add("interpretation", sentence_summary(paragraph_text, 4), child, level)
                    for nested in child.xpath("./fig|./table-wrap|./sec"):
                        if local_name(nested) == "sec":
                            visit(etree.Element("wrapper", children=[]), level + 1)
                        elif local_name(nested) == "fig":
                            label_nodes = nested.xpath("./label")
                            caption_nodes = nested.xpath("./caption")
                            label = clean_text(label_nodes[0]) if label_nodes else "Figure"
                            caption = clean_text(caption_nodes[0]) if caption_nodes else ""
                            graphics = nested.xpath(".//graphic")
                            href = graphics[0].get("{http://www.w3.org/1999/xlink}href", "") if graphics else ""
                            add("figure", f"{label}. {caption}", nested, level, {"href": href})
                    for nested_sec in child.xpath("./sec"):
                        visit(nested_sec, level + 1)
                else:
                    visit(child, level + 1)

    body_nodes = root.xpath("//body")
    if body_nodes:
        if mode == "interpretation":
            for sec in body_nodes[0].xpath("./sec"):
                title_nodes = sec.xpath("./title")
                title = clean_text(title_nodes[0]) if title_nodes else "Section"
                add("section", title, title_nodes[0] if title_nodes else sec, 2)
                direct = " ".join(clean_text(p) for p in sec.xpath("./p"))
                nested = " ".join(clean_text(p) for p in sec.xpath("./sec/p"))
                summary_source = sentence_summary((direct + " " + nested).strip(), 5)
                if summary_source:
                    add("interpretation", summary_source, sec, 2)
                for fig in sec.xpath(".//fig"):
                    label_nodes = fig.xpath("./label")
                    caption_nodes = fig.xpath("./caption")
                    label = clean_text(label_nodes[0]) if label_nodes else "Figure"
                    caption = clean_text(caption_nodes[0]) if caption_nodes else ""
                    graphics = fig.xpath(".//graphic")
                    href = graphics[0].get("{http://www.w3.org/1999/xlink}href", "") if graphics else ""
                    add("figure", f"{label}. {caption}", fig, 2, {"href": href})
        else:
            visit(body_nodes[0], 2)

    refs = root.xpath("//ref-list/ref")
    if refs:
        counter += 1
        blocks.append({"id": f"b{counter:04d}", "type": "references", "source": "\n".join(clean_text(r) for r in refs), "xpath": xpath_of(refs[0]), "level": 2})
    return blocks


def prepare_images(blocks: list[dict], extracted: Path, assets_dir: Path) -> None:
    figures_dir = assets_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    used = set()
    for block in blocks:
        if block["type"] != "figure":
            continue
        found = find_graphic(extracted, block.get("href", ""))
        if found is None:
            block["asset"] = None
            continue
        name = found.name
        if name in used:
            name = f"{block['id']}_{name}"
        used.add(name)
        destination = figures_dir / name
        shutil.copy2(found, destination)
        block["asset"] = f"assets/figures/{name}"


def write_markdown(paper: Paper, meta: dict, title_zh: str, blocks: list[dict], translator: Translator, out_dir: Path) -> None:
    mode_note = "逐段中英对照机器辅助译读版" if paper.mode == "full" else "依据正式论文制作的逐节中文深度解读版（受 CC BY-NC-ND 限制，不是全文翻译）"
    lines = [
        f"# {title_zh}",
        "",
        f"> 原题：{meta['title']}",
        f"> 作者：{', '.join(meta['authors'])}",
        f"> 期刊：{meta['journal']}（{meta['year']}）",
        f"> DOI：https://doi.org/{paper.doi}",
        f"> 许可：{paper.license}",
        f"> 版本说明：{mode_note}；非官方译文。",
        "",
    ]
    for block in blocks:
        kind, source = block["type"], block["source"]
        if kind == "section":
            level = max(2, min(int(block.get("level", 2)), 5))
            lines.extend([f"{'#' * level} {translator.get(source)} / {source}", ""])
        elif kind in {"paragraph", "interpretation"}:
            if kind == "interpretation":
                lines.extend(["**中文解读**", "", translator.get(source), "", "<details><summary>用于解读的原文要点</summary>", "", source, "", "</details>", ""])
            else:
                lines.extend([f"**EN · {block['id']}**", "", source, "", f"**中文 · {block['id']}**", "", translator.get(source), ""])
        elif kind == "figure":
            asset = block.get("asset")
            if asset:
                lines.extend([f"![{html.escape(source)}]({asset})", ""])
            lines.extend([f"**Figure source · {block['id']}** {source}", "", f"**中文图注 · {block['id']}** {translator.get(source)}", ""])
        elif kind == "table":
            lines.extend([f"**Table source · {block['id']}**", "", "```text", source, "```", "", f"**中文表格译读 · {block['id']}**", "", translator.get(source), ""])
        elif kind == "list":
            items = source.splitlines()
            translated = translator.get(source).splitlines()
            lines.extend([f"**EN · {block['id']}**", ""] + [f"- {x}" for x in items] + ["", f"**中文 · {block['id']}**", ""] + [f"- {x}" for x in translated] + [""])
        elif kind == "formula":
            lines.extend([f"**公式 · {block['id']}** `{source}`", ""])
        elif kind == "references":
            lines.extend(["## References（原文参考文献，不翻译）", "", "```text", source, "```", ""])
    (out_dir / "paper.md").write_text("\n".join(lines), encoding="utf-8")


def write_source_map(paper: Paper, blocks: list[dict], out_dir: Path) -> None:
    payload = {
        "paper": paper.key,
        "pmcid": paper.pmcid,
        "doi": paper.doi,
        "license": paper.license,
        "translation_mode": paper.mode,
        "blocks": [
            {
                "block_id": block["id"],
                "type": block["type"],
                "source_locator": f"{paper.xml_name}:{block['xpath']}",
                "asset": block.get("asset"),
                "translated": block["type"] != "references" and not (paper.mode == "interpretation" and block["type"] == "references"),
            }
            for block in blocks
        ],
    }
    (out_dir / "source_map.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_notes(paper: Paper, meta: dict, out_dir: Path) -> None:
    if paper.mode == "full":
        copyright_note = "本译读版依据 CC BY 4.0 制作；保留原作者、题名、期刊、DOI 与许可信息，并明确标注译文属于改编内容。"
    else:
        copyright_note = "正式期刊版采用 CC BY-NC-ND 4.0。为遵守禁止改编条款，本目录不提供逐字全文翻译，仅提供逐节中文解读；原始 PDF 单独原样保存。"
    lines = [
        "# Translation notes",
        "",
        f"- Paper: {meta['title']}",
        f"- DOI: https://doi.org/{paper.doi}",
        f"- License: {paper.license}",
        f"- Copyright handling: {copyright_note}",
        "- Status: 机器辅助初译/解读，非官方译文；用于科研阅读，投稿引用时应以英文原文为准。",
        "- Alignment: `paper.md` 中的块编号与 `source_map.json` 一一对应。",
        "- Figures: 图像来自开放获取附件包；图注与相邻正文按 JATS 顺序放置。",
        "",
        "## Terminology ledger",
        "",
    ]
    lines.extend([f"- {en} → {zh}" for en, zh in TERMS.items()])
    (out_dir / "translation_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_para(text: str) -> str:
    text = html.escape(text).replace("\n", "<br/>")
    return text


def write_pdf(paper: Paper, meta: dict, title_zh: str, blocks: list[dict], translator: Translator, out_dir: Path) -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_bold = Path(r"C:\Windows\Fonts\msyhbd.ttc")
    pdfmetrics.registerFont(TTFont("MSYH", str(font_path), subfontIndex=0))
    if font_bold.exists():
        pdfmetrics.registerFont(TTFont("MSYH-Bold", str(font_bold), subfontIndex=0))
    else:
        pdfmetrics.registerFont(TTFont("MSYH-Bold", str(font_path), subfontIndex=0))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CJKTitle", parent=styles["Title"], fontName="MSYH-Bold", fontSize=18, leading=26, alignment=TA_CENTER, spaceAfter=12)
    h_style = ParagraphStyle("CJKHeading", parent=styles["Heading2"], fontName="MSYH-Bold", fontSize=13, leading=19, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#18324A"))
    body_style = ParagraphStyle("CJKBody", parent=styles["BodyText"], fontName="MSYH", fontSize=9.3, leading=15, spaceAfter=7, wordWrap="CJK")
    caption_style = ParagraphStyle("CJKCaption", parent=body_style, fontSize=8.2, leading=12, textColor=colors.HexColor("#4A5560"), leftIndent=4 * mm, rightIndent=4 * mm)
    meta_style = ParagraphStyle("CJKMeta", parent=body_style, fontSize=8.5, leading=13, textColor=colors.HexColor("#5E6770"))
    note_style = ParagraphStyle("CJKNote", parent=body_style, backColor=colors.HexColor("#EEF4F7"), borderPadding=7, borderColor=colors.HexColor("#BCD0DB"), borderWidth=0.5)

    filename = f"{paper.order:02d}_{paper.key}_中文{'译读' if paper.mode == 'full' else '深度解读'}.pdf"
    document = SimpleDocTemplate(str(out_dir / filename), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title=title_zh, author=", ".join(meta["authors"]))
    story = [Paragraph(safe_para(title_zh), title_style), Paragraph(safe_para(meta["title"]), meta_style), Spacer(1, 3 * mm)]
    story.append(Paragraph(safe_para(f"DOI: {paper.doi}　许可: {paper.license}　非官方机器辅助译读版"), note_style))
    story.append(Spacer(1, 4 * mm))
    for block in blocks:
        kind, source = block["type"], block["source"]
        if kind == "section":
            story.append(Paragraph(safe_para(translator.get(source)), h_style))
        elif kind in {"paragraph", "interpretation"}:
            story.append(Paragraph(safe_para(translator.get(source)), body_style))
        elif kind == "figure":
            asset = block.get("asset")
            if asset:
                path = out_dir / asset
                try:
                    image = Image(str(path))
                    max_w, max_h = 165 * mm, 105 * mm
                    scale = min(max_w / image.imageWidth, max_h / image.imageHeight, 1.0)
                    image.drawWidth = image.imageWidth * scale
                    image.drawHeight = image.imageHeight * scale
                    story.extend([Spacer(1, 2 * mm), image, Spacer(1, 1 * mm)])
                except Exception:
                    pass
            story.append(Paragraph(safe_para(translator.get(source)), caption_style))
        elif kind == "table":
            story.append(Paragraph(safe_para(translator.get(source)), caption_style))
        elif kind == "list":
            story.append(Paragraph(safe_para(translator.get(source)), body_style))
        elif kind == "formula":
            story.append(Paragraph(safe_para(source), caption_style))
        elif kind == "references":
            story.append(PageBreak())
            story.append(Paragraph("参考文献见英文原文", h_style))
            story.append(Paragraph("为避免文献题名与作者信息在机器翻译中失真，本中文 PDF 不重复参考文献表；请查阅同目录保存的英文原文 PDF 或 paper.md。", body_style))
    document.build(story)


def process(paper: Paper) -> None:
    print(f"Processing {paper.key}", flush=True)
    parser = etree.XMLParser(resolve_entities=False, recover=True, huge_tree=True)
    root = etree.parse(str(SOURCE / paper.xml_name), parser).getroot()
    meta = metadata(root)
    out_dir = OUTPUT / f"{paper.order:02d}_{paper.key}"
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = unpack_assets(paper, out_dir)
    blocks = collect_blocks(root, paper.mode)
    prepare_images(blocks, extracted, out_dir / "assets")
    translator = Translator(out_dir / "translation_cache.json")
    texts = [meta["title"]] + [block["source"] for block in blocks if block["type"] not in {"formula", "references"}]
    translator.translate_many(texts)
    title_zh = translator.get(meta["title"])
    write_markdown(paper, meta, title_zh, blocks, translator, out_dir)
    write_source_map(paper, blocks, out_dir)
    write_notes(paper, meta, out_dir)
    write_pdf(paper, meta, title_zh, blocks, translator, out_dir)
    shutil.rmtree(extracted, ignore_errors=True)
    print(f"Completed {paper.key}: {len(blocks)} blocks", flush=True)


def write_catalog() -> None:
    rows = [
        "# 五个正式基线方法论文",
        "",
        "本目录保存五个基线方法对应的开放获取论文原文，以及机器辅助中文译读文件。中文文件不是官方译文；论文引用、公式和定量结论请以英文原文为准。",
        "",
        "| 方法 | 正式论文 | DOI | 许可 | 中文版本 |",
        "|---|---|---|---|---|",
    ]
    for paper in PAPERS:
        mode = "逐段中英对照全文译读" if paper.mode == "full" else "逐节中文深度解读（不提供全文翻译）"
        rows.append(f"| {paper.key} | `{paper.pdf_name}` | [{paper.doi}](https://doi.org/{paper.doi}) | {paper.license} | {mode} |")
    rows.extend([
        "",
        "## 版权说明",
        "",
        "- POLYGON、REINVENT4、DrugEx v2 与 Mothra 的正式论文采用 CC BY 4.0；译读版注明来源、许可和改编性质。",
        "- MO-LSO 的正式期刊版采用 CC BY-NC-ND 4.0，禁止发布改编版本。因此这里只保存原文 PDF，并提供不替代原文的逐节中文解读。",
        "- `_source` 目录用于保存 Europe PMC 的结构化正文与附件来源，便于复核；`source_map.json` 保存段落级来源定位。",
    ])
    (BASE / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_catalog()
    for paper in PAPERS:
        process(paper)


if __name__ == "__main__":
    main()
