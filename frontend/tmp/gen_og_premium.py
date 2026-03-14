from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

W, H = 1200, 630
img = Image.new('RGB', (W, H), '#1C1640')
px = img.load()

# Deep violet -> lavender smooth gradient
c1 = (23, 18, 56)
c2 = (48, 33, 104)
c3 = (124, 110, 206)
for y in range(H):
    for x in range(W):
        t = (x / W) * 0.58 + (y / H) * 0.42
        if t < 0.60:
            k = t / 0.60
            r = int(c1[0] * (1-k) + c2[0] * k)
            g = int(c1[1] * (1-k) + c2[1] * k)
            b = int(c1[2] * (1-k) + c2[2] * k)
        else:
            k = (t - 0.60) / 0.40
            r = int(c2[0] * (1-k) + c3[0] * k)
            g = int(c2[1] * (1-k) + c3[1] * k)
            b = int(c2[2] * (1-k) + c3[2] * k)
        px[x, y] = (r, g, b)

# subtle geometric light (no boxes)
geo = Image.new('RGBA', (W, H), (0,0,0,0))
g = ImageDraw.Draw(geo)
g.polygon([(0,0),(520,0),(290,370)], fill=(185,172,255,26))
g.polygon([(1200,0),(1200,360),(860,120)], fill=(214,204,255,22))
g.polygon([(0,630),(450,630),(140,420)], fill=(178,165,255,20))
geo = geo.filter(ImageFilter.GaussianBlur(30))
img = Image.alpha_composite(img.convert('RGBA'), geo).convert('RGB')
d = ImageDraw.Draw(img)

# metallic silver heart based on favicon motif
heart_mask = Image.new('L', (W, H), 0)
hm = ImageDraw.Draw(heart_mask)
ox, oy = 272, 208
s = 1.50
hm.ellipse((int(ox-78*s), int(oy-74*s), int(ox-12*s), int(oy-8*s)), fill=255)
hm.ellipse((int(ox-18*s), int(oy-74*s), int(ox+48*s), int(oy-8*s)), fill=255)
hm.polygon([
    (int(ox-78*s), int(oy-42*s)),
    (int(ox-15*s), int(oy+74*s)),
    (int(ox+48*s), int(oy-42*s))
], fill=255)

silver = Image.new('RGBA', (W, H), (0,0,0,0))
spx = silver.load()
for y in range(H):
    for x in range(W):
        if heart_mask.getpixel((x,y)) > 0:
            t = (x - (ox-90*s)) / max(1, int(170*s))
            t = max(0.0, min(1.0, t))
            base = int(184 + 46*t)
            spx[x,y] = (base, base+4, base+12, 255)

hl = Image.new('RGBA', (W, H), (0,0,0,0))
hld = ImageDraw.Draw(hl)
hld.ellipse((int(ox-62*s), int(oy-63*s), int(ox-26*s), int(oy-31*s)), fill=(255,255,255,88))
hld.ellipse((int(ox+2*s), int(oy-58*s), int(ox+30*s), int(oy-34*s)), fill=(255,255,255,72))

shadow = Image.new('RGBA', (W, H), (0,0,0,0))
sd = ImageDraw.Draw(shadow)
sd.ellipse((int(ox-92*s), int(oy-76*s), int(ox+60*s), int(oy+82*s)), fill=(15,10,36,120))
shadow = shadow.filter(ImageFilter.GaussianBlur(16))
img = Image.alpha_composite(img.convert('RGBA'), shadow)
img = Image.alpha_composite(img, silver)
img = Image.alpha_composite(img, hl).convert('RGB')
d = ImageDraw.Draw(img)

# sleek smooth PPG waveform across lower-center
points = []
start_x, end_x = 120, 1080
base_y = 430
for x in range(start_x, end_x+1, 4):
    y = base_y + 5 * math.sin((x-start_x)/70.0)
    if 500 <= x <= 560:
        peak_t = 1 - abs(x-530)/30
        y -= max(0, peak_t) * 84
    elif 560 < x <= 595:
        dip_t = 1 - abs(x-578)/18
        y += max(0, dip_t) * 44
    elif 595 < x <= 645:
        rec_t = 1 - abs(x-620)/25
        y -= max(0, rec_t) * 26
    points.append((x, int(y)))

d.line(points, fill=(245,244,255), width=5)
d.line(points, fill=(198,189,255), width=2)

# Fonts (force Korean-capable)
def f(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

font_title = f([
    r'C:\Windows\Fonts\segoeuib.ttf',
    r'C:\Windows\Fonts\malgunbd.ttf'
], 84)
font_sub = f([
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\segoeui.ttf'
], 36)
font_chip = f([
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\malgun.ttf'
], 21)
font_url = f([
    r'C:\Windows\Fonts\segoeui.ttf',
    r'C:\Windows\Fonts\malgun.ttf'
], 27)

# Text block, no clumsy box
d.text((430, 182), 'VisiVital', font=font_title, fill=(249, 250, 255))
d.text((432, 278), 'PPG 건강 참고 모니터링', font=font_sub, fill=(236, 233, 252))

# Premium pills
chips = ['Real-time', 'Non-contact', 'AI Insights']
x = 432
y = 333
for chip in chips:
    tw = d.textlength(chip, font=font_chip)
    pw = int(tw + 28)
    d.rounded_rectangle((x, y, x+pw, y+34), radius=17, fill=(249,247,255), outline=(229,221,255), width=1)
    d.text((x+14, y+7), chip, font=font_chip, fill=(71, 56, 126))
    x += pw + 12

# URL at bottom center
url = 'yonseihci.kro.kr'
uw = d.textlength(url, font=font_url)
d.text(((W-uw)/2, 564), url, font=font_url, fill=(232, 229, 252))

out_main = r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image.png'
out_v2 = r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image-v2.png'
img.save(out_main, 'PNG', optimize=True)
img.save(out_v2, 'PNG', optimize=True)
print('OG_PREMIUM_DONE')
