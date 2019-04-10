import sys
import os
from PIL import Image, ImageDraw, ImageFont

datasets = ['s8_gg', 'h_qq', 'qx_qg', 'cp_qq', 'h_gg', 'zp_qq']
datasets_c = ['s8_gg_rot_charged', 'h_qq_rot_charged', 'qx_qg_rot_charged', 'cp_qq_rot_charged', 'h_gg_rot_charged', 'zp_qq_rot_charged']

# for combine, 0 is pcc, 1 is roc, 2 is sic
def u_tri(using_charged, combine):
    images = []
    for i in range(6):
        for j in range(6):
            if i >= j:
                continue

            if combine == 0:
                f_path = pcc_path
            elif combine == 1:
                f_path = roc_path
            else:
                f_path = sic_path

            if using_charged:
                path = c_path(f_path, datasets_c[i], datasets_c[j], True)
            else:
                path = c_path(f_path, datasets[i], datasets[j])

            images.append(Image.open(path))

    width, height = images[0].size
    txt_offset = 200
    comb_im = Image.new('RGB', (width * 6 + txt_offset, height * 6 + txt_offset), color=(255,255,255))

    draw = ImageDraw.Draw(comb_im)
    font = ImageFont.truetype("../../Roboto-Black.ttf", 50)
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

    path = 'final_curves/combined/all_'
    if combine == 0:
        path = path + 'pcc'
    elif combine == 1:
        path = path + 'roc'
    else:
        path = path + 'sic'

    if using_charged:
        path = path + '_charged'
    path = path + '.png'
    comb_im.save(path)

def c_path(path_f, sig, bg, charged = False):
    if os.path.isfile(path_f(sig, bg)):
        return path_f(sig, bg)
    else:
        return path_f(bg, sig)

def pcc_path(sig, bg):
    return 'final_curves/pearsons/truths/' + sig + '_vs_' + bg + '_pearson_truth.png'
def sic_path(sig, bg):
    return 'final_curves/sic_' + sig + '_vs_' + bg + '.png'
def roc_path(sig, bg):
    return 'final_curves/roc_' + sig + '_vs_' + bg + '.png'
def img_path(sig):
    return 'final_curves/Average_' + sig + '.png'

def all_img(charged):
    images = []
    for i in range(6):
        if charged:
            images.append(Image.open(img_path(datasets_c[i])))
        else:
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

    if charged:
        comb_im.save('final_curves/combined/all_img_charged.png')
    else:
        comb_im.save('final_curves/combined/all_img.png')

def cp_main():
    for i in [False, True]:
        all_img(i)
        for j in range(3):
            u_tri(i, j)

if __name__ == '__main__':
  cp_main()