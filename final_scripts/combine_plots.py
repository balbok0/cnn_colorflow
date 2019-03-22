import sys
import os
from PIL import Image, ImageDraw, ImageFont

#datasets = ['s8_gg', 'h_qq', 'qx_qg', 'cp_qq', 'h_gg', 'zp_qq']
datasets = ['h_qq_rot_charged', 'h_gg_rot_charged', 'cp_qq_rot_charged', 'qx_qg_rot_charged', 's8_gg_rot_charged', 'zp_qq_rot_charged']

def main():
    images = []
    for i in range(6):
        for j in range(6):
            if i >= j:
                continue

            sig = datasets[i]
            bg = datasets[j]
            path = c_path(pcc_path, sig, bg)
            images.append(Image.open(path))

    width, height = images[0].size
    txt_offset = 200
    comb_im = Image.new('RGB', (width * 6 + txt_offset, height * 6 + txt_offset), color=(255,255,255))

    draw = ImageDraw.Draw(comb_im)
    font = ImageFont.truetype("../../Roboto-Black.ttf", 120)
    for i in range(6):
        center_offset = 180
        draw.text((txt_offset + center_offset + width*i, 0), datasets[i], (0,0,0),font=font)
        draw.text((0, txt_offset + center_offset + height*i), datasets[i], (0,0,0),font=font)

    x_offset = 0
    y_offset = 0
    for im in images:
        comb_im.paste(im, (x_offset * width + txt_offset + width, y_offset * height + txt_offset))
        x_offset += 1
        if x_offset >= 5:
            y_offset += 1
            x_offset = y_offset

    comb_im.save('final_curves/all_pcc.png')

def c_path(path_f, sig, bg):
    if os.path.isfile(path_f(sig, bg)):
        return path_f(sig, bg)
    else:
        return path_f(bg, sig)

def pcc_path(sig, bg):
    return 'final_curves/pearsons/truths/' + sig + ' vs ' + bg + ' charged_pearson_truth.png'
def sic_path(sig, bg):
    return 'final_curves/sic_' + sig + ' vs ' + bg + ' charged.png'
def roc_path(sig, bg):
    return 'final_curves/roc_' + sig + ' vs ' + bg + ' charged.png'
def img_path(sig):
    return 'final_curves/Average_' + sig + '.png'

def main2():
    images = []
    for i in range(6):
        images.append(Image.open(img_path(datasets[i])))

    width, height = images[0].size
    comb_im = Image.new('RGB', (width * 3, height * 2), color=(255,255,255))

    x_offset = 0
    y_offset = 0
    for im in images:
        comb_im.paste(im, (x_offset * width, y_offset * height))
        x_offset += 1
        if x_offset >= 3:
            y_offset += 1
            x_offset = 0

    comb_im.save('final_curves/all_img.png')

if __name__ == '__main__':
  main2()