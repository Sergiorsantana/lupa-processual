"""Apoio visual: cores por severidade e destaque dos achados sobre a imagem da página."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

SEV_COLOR = {"critica": "#d03b3b", "alta": "#ec835a", "media": "#fab219", "baixa": "#6da7ec"}
VERDICT_COLOR = {"CONFORME": "#0ca30c", "SUSPEITO": "#fab219", "NÃO CONFORME": "#d03b3b"}
SEV_LABEL = {"critica": "Crítica", "alta": "Alta", "media": "Média", "baixa": "Baixa"}


def _rgb(hexcolor: str) -> tuple:
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def highlight_page(image_path: str, findings: list, page_w: float, only_ids: set | None = None) -> Image.Image:
    """Desenha caixas numeradas sobre a página. bbox está em unidades da página; a escala vem da largura da imagem."""
    img = Image.open(image_path).convert("RGB")
    scale = img.width / page_w if page_w else 1.0
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    try:
        font = ImageFont.load_default(size=max(14, int(img.width / 55)))
    except TypeError:
        font = ImageFont.load_default()
    for f in findings:
        if not f.bbox or f.status != "ativo":
            continue
        if only_ids is not None and f.id not in only_ids:
            continue
        x0, y0, x1, y1 = [v * scale for v in f.bbox]
        # caixas minúsculas (fonte de 1 pt) ficam invisíveis, então ganham uma folga mínima
        pad = max(3, int(img.width / 300))
        box = [max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, img.width - 1), min(y1 + pad, img.height - 1)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        r, g, b = _rgb(SEV_COLOR.get(f.severity, "#555555"))
        d.rectangle(box, fill=(r, g, b, 55), outline=(r, g, b, 255), width=max(2, int(img.width / 450)))
        label = str(f.num)
        tw = d.textlength(label, font=font)
        th = font.size if hasattr(font, "size") else 14
        # etiqueta à direita da caixa, centralizada na altura; se não couber, vai para dentro da borda direita
        cy = (box[1] + box[3]) / 2
        tx = box[2] + 4 if box[2] + tw + 14 < img.width else box[2] - tw - 10
        tag = [tx, max(cy - (th + 6) / 2, 0), tx + tw + 10, max(cy - (th + 6) / 2, 0) + th + 6]
        d.rectangle(tag, fill=(r, g, b, 255))
        d.text((tag[0] + 5, tag[1] + 2), label, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
