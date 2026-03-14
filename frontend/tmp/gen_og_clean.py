from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

W, H = 1200, 630
S = 2
w2, h2 = W*S, H*S

c1 = (92, 124, 250)
c2 = (116, 143, 252)
img = Image.new('RGB', (w2, h2), c1)
p = img.load()

for y in range(h2):
    for x in range(w2):
        t = (x / w2) * 0.62 + (y / h2) * 0.38
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        p[x, y] = (r, g, b)

overlay = Image.new('RGBA', (w2, h2), (0,0,0,0))
od = ImageDraw.Draw(overlay)
od.ellipse((int(w2*0.58), int(-h2*0.15), int(w2*1.15), int(h2*0.45)), fill=(255,255,255,36))
overlay = overlay.filter(ImageFilter.GaussianBlur(48))
img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
d = ImageDraw.Draw(img)

# standalone heart mark only (favicon heart silhouette)
hx = int(w2 * 0.22)
hy = int(h2 * 0.295)
heart_scale = 2.05 * S

heart = Image.new('RGBA', (w2, h2), (0, 0, 0, 0))
hd = ImageDraw.Draw(heart)

def cubic_bezier(p0, p1, p2, p3, steps=40):
    points = []
    for index in range(steps + 1):
        t = index / steps
        mt = 1.0 - t
        x = (
            mt * mt * mt * p0[0]
            + 3 * mt * mt * t * p1[0]
            + 3 * mt * t * t * p2[0]
            + t * t * t * p3[0]
        )
        y = (
            mt * mt * mt * p0[1]
            + 3 * mt * mt * t * p1[1]
            + 3 * mt * t * t * p2[1]
            + t * t * t * p3[1]
        )
        points.append((x, y))
    return points

# Favicon heart path (translated to polygon samples)
# M0,45 L-7,-7 C-28,-72 -100,-72 -100,-14 C-100,44 0,115 0,115
# C0,115 100,44 100,-14 C100,-72 28,-72 7,-7 Z
base = []
base.append((0.0, 45.0))
base.append((-7.0, -7.0))

curve1 = cubic_bezier((-7.0, -7.0), (-28.0, -72.0), (-100.0, -72.0), (-100.0, -14.0), 36)
curve2 = cubic_bezier((-100.0, -14.0), (-100.0, 44.0), (0.0, 115.0), (0.0, 115.0), 36)
curve3 = cubic_bezier((0.0, 115.0), (0.0, 115.0), (100.0, 44.0), (100.0, -14.0), 36)
curve4 = cubic_bezier((100.0, -14.0), (100.0, -72.0), (28.0, -72.0), (7.0, -7.0), 36)

base.extend(curve1[1:])
base.extend(curve2[1:])
base.extend(curve3[1:])
base.extend(curve4[1:])

heart_points = []
for x, y in base:
    px = int(hx + x * heart_scale)
    py = int(hy + y * heart_scale)
    heart_points.append((px, py))

hd.polygon(heart_points, fill=(246, 248, 255, 246))

# subtle highlight + glow for depth
glow = Image.new('RGBA', (w2, h2), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.ellipse((hx - int(170 * S), hy - int(175 * S), hx + int(170 * S), hy + int(190 * S)), fill=(255, 255, 255, 28))
glow = glow.filter(ImageFilter.GaussianBlur(int(11 * S)))

hl = Image.new('RGBA', (w2, h2), (0, 0, 0, 0))
hld = ImageDraw.Draw(hl)
hld.ellipse((hx - int(54 * S), hy - int(84 * S), hx - int(2 * S), hy - int(34 * S)), fill=(255, 255, 255, 44))
hl = hl.filter(ImageFilter.GaussianBlur(int(2 * S)))

heart = heart.filter(ImageFilter.GaussianBlur(int(0.8 * S)))
img = Image.alpha_composite(img.convert('RGBA'), glow)
img = Image.alpha_composite(img, heart)
img = Image.alpha_composite(img, hl).convert('RGB')
d = ImageDraw.Draw(img)

wave = [
    (int(w2*0.07), int(h2*0.70)),
    (int(w2*0.15), int(h2*0.70)),
    (int(w2*0.20), int(h2*0.70)),
    (int(w2*0.23), int(h2*0.59)),
    (int(w2*0.265), int(h2*0.82)),
    (int(w2*0.30), int(h2*0.55)),
    (int(w2*0.335), int(h2*0.79)),
    (int(w2*0.37), int(h2*0.63)),
    (int(w2*0.405), int(h2*0.70)),
    (int(w2*0.93), int(h2*0.70)),
]
d.line(wave, fill=(255,255,255,242), width=int(8*S), joint='curve')

def get_font(paths, size):
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

font_title = get_font([r'C:\Windows\Fonts\segoeuib.ttf', r'C:\Windows\Fonts\malgunbd.ttf'], int(62*S))
font_sub = get_font([r'C:\Windows\Fonts\malgun.ttf', r'C:\Windows\Fonts\segoeui.ttf'], int(28*S))
font_url = get_font([r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\malgun.ttf'], int(22*S))

tx = int(w2*0.40)
d.text((tx, int(h2*0.24)), 'VisiVital', font=font_title, fill=(255,255,255,248))
d.text((tx, int(h2*0.39)), 'rPPG Health Reference Monitoring', font=font_sub, fill=(246,248,255,236))

chips = ['Real-time Vital Insight', 'Non-contact', 'AI Estimation']
chip_font_size = int(15 * S)
chip_font = get_font([r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\malgun.ttf'], chip_font_size)
chip_padding_x = int(12 * S)
chip_gap = int(9 * S)
chip_height = int(28 * S)
chip_y = int(h2 * 0.50)
chip_start_x = tx
chip_end_x = int(w2 * 0.93)

def total_chip_width(font):
    total = 0
    for index, chip in enumerate(chips):
        bbox = d.textbbox((0, 0), chip, font=font)
        text_width = bbox[2] - bbox[0]
        total += text_width + chip_padding_x * 2
        if index < len(chips) - 1:
            total += chip_gap
    return total

while total_chip_width(chip_font) > (chip_end_x - chip_start_x) and chip_font_size > int(11 * S):
    chip_font_size -= 1
    chip_font = get_font([r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\malgun.ttf'], chip_font_size)

x = chip_start_x
for chip in chips:
    bbox = d.textbbox((0, 0), chip, font=chip_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    chip_width = text_width + chip_padding_x * 2

    d.rounded_rectangle(
        (x, chip_y, x + chip_width, chip_y + chip_height),
        radius=int(14 * S),
        fill=(248, 250, 255),
        outline=(231, 236, 255),
        width=1,
    )

    text_x = x + chip_padding_x
    text_y = chip_y + (chip_height - text_height) // 2 - 1
    d.text((text_x, text_y), chip, font=chip_font, fill=(70, 88, 168))
    x += chip_width + chip_gap

url = 'yonseihci.kro.kr'
uw = d.textlength(url, font=font_url)
d.text(((w2-uw)/2, int(h2*0.89)), url, font=font_url, fill=(244,247,255,220))

out = img.resize((W, H), Image.LANCZOS)
main_path = r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image.png'
v2_path = r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image-v2.png'
out.save(main_path, 'PNG', optimize=True)
out.save(v2_path, 'PNG', optimize=True)
print('OG_UPDATED_NO_TRIANGLE')
