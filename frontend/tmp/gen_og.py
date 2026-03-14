from PIL import Image, ImageDraw, ImageFont
W,H = 1200,630
img = Image.new('RGB',(W,H),(98,126,246))
d = ImageDraw.Draw(img)
# wave line
for x in range(0,W,2):
    y = int(H*0.62 + 18*((x/42.0)%2-1))
    d.line([(x,y),(x+2,int(H*0.62 + 18*(((x+2)/42.0)%2-1)))], fill=(182,198,255), width=3)
# heart shape (simple)
d.ellipse((80,70,260,250), fill=(130,155,255))
d.ellipse((200,70,380,250), fill=(130,155,255))
d.polygon([(80,170),(290,430),(500,170)], fill=(130,155,255))

def get_font(candidates, size):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

font_title = get_font([
    r'C:\Windows\Fonts\malgunbd.ttf',
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\arialbd.ttf'
], 88)
font_sub = get_font([
    r'C:\Windows\Fonts\malgun.ttf',
    r'C:\Windows\Fonts\arial.ttf'
], 42)
font_url = get_font([
    r'C:\Windows\Fonts\arial.ttf',
    r'C:\Windows\Fonts\malgun.ttf'
], 36)

d.text((330,170), 'VisiVital', font=font_title, fill=(240,245,255))
d.text((330,280), 'rPPG 건강 참고 모니터링', font=font_sub, fill=(236,240,250))
d.text((330,350), 'yonseihci.kro.kr', font=font_url, fill=(220,233,255))
img.save(r'e:\Yonsei AI\WorkSpace\Web\_visivital_publish_src\frontend\public\og-image.png', 'PNG', optimize=True)
print('OG_IMAGE_WRITTEN')
