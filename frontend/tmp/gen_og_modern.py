from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

W, H = 1200, 630
img = Image.new('RGB', (W, H), '#0b1020')
p = img.load()

# Smooth diagonal gradient background
c1 = (11, 16, 32)
c2 = (48, 86, 216)
c3 = (111, 183, 255)
for y in range(H):
    for x in range(W):
        t = (x / W) * 0.65 + (y / H) * 0.35
        if t < 0.55:
            k = t / 0.55
            r = int(c1[0] * (1-k) + c2[0] * k)
            g = int(c1[1] * (1-k) + c2[1] * k)
            b = int(c1[2] * (1-k) + c2[2] * k)
        else:
            k = (t - 0.55) / 0.45
            r = int(c2[0] * (1-k) + c3[0] * k)
            g = int(c2[1] * (1-k) + c3[1] * k)
            b = int(c2[2] * (1-k) + c3[2] * k)
        p[x, y] = (r, g, b)

# Soft radial glow accents
for cx, cy, radius, col in [
    (980, 120, 220, (157, 214, 255)),
    (220, 520, 280, (88, 127, 255)),
]:
    glow = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(*col, 75))
    glow = glow.filter(ImageFilter.GaussianBlur(55))
    img = Image.alpha_composite(img.convert('RGBA'), glow).convert('RGB')

d = ImageDraw.Draw(img)

# Glass card panel
panel = Image.new('RGBA', (W, H), (0,0,0,0))
pd = ImageDraw.Draw(panel)
pd.rounded_rectangle((70, 70, 1130, 560), radius=40, fill=(255, 255, 255, 34), outline=(255, 255, 255, 85), width=2)
panel = panel.filter(ImageFilter.GaussianBlur(0.5))
img = Image.alpha_composite(img.convert('RGBA'), panel).convert('RGB')
d = ImageDraw.Draw(img)

# Heart icon + pulse line
heart_layer = Image.new('RGBA', (W, H), (0,0,0,0))
hd = ImageDraw.Draw(heart_layer)
hd.ellipse((130, 170, 215, 255), fill=(173, 219, 255, 210))
hd.ellipse((195, 170, 280, 255), fill=(173, 219, 255, 210))
hd.polygon([(130, 212), (238, 345), (346, 212)], fill=(173, 219, 255, 210))
heart_layer = heart_layer.filter(ImageFilter.GaussianBlur(0.4))
img = Image.alpha_composite(img.convert('RGBA'), heart_layer).convert('RGB')
d = ImageDraw.Draw(img)

points = []
base_y = 365
for x in range(120, 1080, 8):
    y = base_y + int(8*math.sin(x/36))
    if 360 < x < 430:
        y -= int((1-abs(395-x)/35)*70)
    if 430 <= x < 470:
        y += int((1-abs(450-x)/20)*45)
    if 470 <= x < 520:
        y -= int((1-abs(495-x)/25)*28)
    points.append((x, y))

d.line(points, fill=(216, 238, 255), width=6)
d.line(points, fill=(132, 204, 255), width=2)

# Typography helpers

def get_font(paths, size):
    for fp in paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()

font_title = get_font([
    r'C:\Windows\Fonts\malgunbd.ttf',
    r'C:\Windows\Fonts\segoeuib.ttf',
    r'C:\Windows\Fonts\arialbd.ttf'
], 88)
font_sub = get_font([
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\arial.ttf'
], 38)
font_chip = get_font([
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\segoeui.ttf'
], 26)
font_url = get_font([
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\malgun.ttf'
], 30)

# Text block
d.text((400, 175), 'VisiVital', font=font_title, fill=(242, 248, 255))
d.text((402, 276), 'rPPG 건강 참고 모니터링', font=font_sub, fill=(226, 238, 253))

# Feature chips
chips = ['Real-time Vital Insight', 'Non-contact', 'AI Estimation']
start_x = 400
chip_y = 338
for chip in chips:
    tw = d.textlength(chip, font=font_chip)
    w = int(tw + 34)
    d.rounded_rectangle((start_x, chip_y, start_x + w, chip_y + 44), radius=18, fill=(255, 255, 255, 42), outline=(255,255,255,90), width=1)
    d.text((start_x + 17, chip_y + 8), chip, font=font_chip, fill=(235, 245, 255))
    start_x += w + 14

# Footer url
d.text((400, 430), 'yonseihci.kro.kr', font=font_url, fill=(210, 232, 255))

out = r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image.png'
img.save(out, 'PNG', optimize=True)
print('OG_REDESIGN_DONE')
