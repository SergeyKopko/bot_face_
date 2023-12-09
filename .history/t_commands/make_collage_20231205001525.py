import os
import random
import uuid
import io

from PIL import Image

from t_commands.db import *
from constants import COUNT_PHOTO_TURBO


def make_collage(
    image_target_byte, message_from_id, user_images_id, width, init_height
):
    """Функция делает коллаж из фоток и возвращает ссылку на коллаж."""
    print(7)

    # images_all = get_image_target_links(user_images_id, db)
    images_all = image_target_byte
    images = images_all
    if len(images_all) >= COUNT_PHOTO_PREM:
        print(9)
        images = images[:100]
    if not images:
        print("Не найдены изображения для создания коллажа! make_collage")
        return False

    margin_size = 2
    img_list = []
    for i in images:
        img_list.append(Image.open(io.BytesIO(i)))
        # img_list.append(Image.open(MINIO_CLIENT.get_object(BUCKET_NAME, i)))
    while True:
        # copy images to images_list
        images_list = images[:]
        coefs_lines = []
        images_line = []
        x = 0
        for n, img_path in enumerate(images_list):
            # get first image and resize to `init_height`
            img = img_list[n]
            img.thumbnail((width, init_height))
            # when `x` will go beyond the `width`, start the next line
            if x > width:
                coefs_lines.append((float(x) / width, images_line))
                images_line = []
                x = 0
            x += img.size[0] + margin_size
            images_line.append(img_path)
        # finally add the last line with images
        coefs_lines.append((float(x) / width, images_line))

        # compact the lines, by reducing the `init_height`, if any with one or less images
        if len(coefs_lines) <= 1:
            break
        if any(map(lambda c: len(c[1]) <= 1, coefs_lines)):
            # reduce `init_height`
            init_height -= 10
        else:
            break

    # get output height
    out_height = 0
    for coef, imgs_line in coefs_lines:
        if imgs_line:
            out_height += int(init_height / coef) + margin_size
    if not out_height:
        print("Height of collage could not be 0!")
        return False

    collage_image = Image.new("RGB", (width, int(out_height)), (35, 35, 35))
    # put images to the collage
    y = 0
    for coef, imgs_line in coefs_lines:
        if imgs_line:
            x = 0
            for img_path in imgs_line:
                img = Image.open(io.BytesIO(img_path))
                # if need to enlarge an image - use `resize`, otherwise use `thumbnail`, it's faster
                k = (init_height / coef) / img.size[1]
                if k > 1:
                    img = img.resize(
                        (int(img.size[0] * k), int(img.size[1] * k)),
                        Image.LANCZOS,
                    )
                else:
                    img.thumbnail(
                        (int(width / coef), int(init_height / coef)),
                        Image.LANCZOS,
                    )
                if collage_image:
                    collage_image.paste(img, (int(x), int(y)))
                x += img.size[0] + margin_size
            y += int(init_height / coef) + margin_size

    if len(images_all) <= COUNT_PHOTO_PREM:
        print(8)
        image_bottom = Image.open("./images/for_collage_100.jpg")
    new_image = Image.new("RGB", (width, int(out_height) + 91), (35, 35, 35))
    new_image.paste(collage_image, (0, 0))
    new_image.paste(
        image_bottom,
        (0, out_height),
    )

    path = f"images/{message_from_id}/"
    if not os.path.exists(f"{path}collage/"):
        os.makedirs(f"{path}collage/")
    photo_name = f"{uuid.uuid4().hex}.jpg"
    new_file_path = os.path.join(f"{path}collage/", photo_name)
    new_image = new_image.convert("RGB")
    new_image.save(new_file_path, quality=80, optimize=True)
    new_image.close()

    MINIO_CLIENT.fput_object(BUCKET_NAME, new_file_path, new_file_path)
    if os.path.isfile(new_file_path):
        os.remove(new_file_path)

    return new_file_path


def make_ref_collage(
    image_target_byte, message_from_id, user_images_id, width, init_height, db
):
    """Функция делает коллаж из пяти фоток и возвращает ссылку на коллаж."""

    images = image_target_byte
    # images = random.sample(images, k=5)
    if not images:
        print("Не найдены изображения для создания коллажа! make_ref_collage")
        return False

    margin_size = 2
    img_list = []
    for i in images:
        img_list.append(Image.open(io.BytesIO(i)))
    while True:
        # copy images to images_list
        images_list = images[:]
        coefs_lines = []
        images_line = []
        x = 0
        for n, img_path in enumerate(images_list):
            # get first image and resize to `init_height`
            img = img_list[n]
            img.thumbnail((width, init_height))
            # when `x` will go beyond the `width`, start the next line
            if x > width:
                coefs_lines.append((float(x) / width, images_line))
                images_line = []
                x = 0
            x += img.size[0] + margin_size
            images_line.append(img_path)
        # finally add the last line with images
        coefs_lines.append((float(x) / width, images_line))

        # compact the lines, by reducing the `init_height`, if any with one or less images
        if len(coefs_lines) <= 1:
            break
        if any(map(lambda c: len(c[1]) <= 1, coefs_lines)):
            # reduce `init_height`
            init_height -= 10
        else:
            break

    # get output height
    out_height = 0
    for coef, imgs_line in coefs_lines:
        if imgs_line:
            out_height += int(init_height / coef) + margin_size
    if not out_height:
        print("Height of collage could not be 0!")
        return False

    collage_image = Image.new("RGB", (width, int(out_height)), (35, 35, 35))
    # put images to the collage
    y = 0
    for coef, imgs_line in coefs_lines:
        if imgs_line:
            x = 0
            for img_path in imgs_line:
                img = Image.open(io.BytesIO(img_path))
                # if need to enlarge an image - use `resize`, otherwise use `thumbnail`, it's faster
                k = (init_height / coef) / img.size[1]
                if k > 1:
                    img = img.resize(
                        (int(img.size[0] * k), int(img.size[1] * k)),
                        Image.LANCZOS,
                    )
                else:
                    img.thumbnail(
                        (int(width / coef), int(init_height / coef)),
                        Image.LANCZOS,
                    )
                if collage_image:
                    collage_image.paste(img, (int(x), int(y)))
                x += img.size[0] + margin_size
            y += int(init_height / coef) + margin_size

    ref_overlay_1 = Image.open("./images/ref_overlay_1.png")
    ref_overlay_2 = Image.open("./images/ref_overlay_2.png")
    ref_overlay_1 = ref_overlay_1.crop((0, 1280 - out_height, width, 1280))
    ref_overlay_2 = ref_overlay_2.resize((width - 100, 116)).convert("RGBA")
    ref_overlay_1 = ref_overlay_1.copy()
    # Настраиваемый параметр альфа-канала (значение от 0 до 255)
    alpha = 40
    ref_overlay_1.putalpha(alpha)
    ref_overlay_1 = ref_overlay_1.convert("RGBA")
    collage_image = collage_image.convert("RGBA")
    collage_image = Image.alpha_composite(collage_image, ref_overlay_1)
    collage_image.paste(ref_overlay_2, (50, out_height - 130), ref_overlay_2)

    path = f"images/{message_from_id}/"
    if not os.path.exists(f"{path}ref_collage/"):
        os.makedirs(f"{path}ref_collage/")
    photo_name = f"{uuid.uuid4().hex}.jpg"
    new_file_path = os.path.join(f"{path}ref_collage/", photo_name)
    collage_image = collage_image.convert("RGB")
    collage_image.save(new_file_path, optimize=True)
    collage_image.close()

    MINIO_CLIENT.fput_object(BUCKET_NAME, new_file_path, new_file_path)
    if os.path.isfile(new_file_path):
        os.remove(new_file_path)

    return new_file_path
