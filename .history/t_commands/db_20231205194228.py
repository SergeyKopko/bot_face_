from datetime import datetime

from constants import *


def get_paid_collage_target_links(user_id, image_tarif, db):
    """
    Возвращает массив ссылок на готовые изображения.
    """

    with db.cursor() as connection:
        with connection.transaction():
            query_select_id = """
            SELECT id 
            FROM user_images 
            WHERE user_id=%s 
            AND image_tarif=%s
            AND status=%s
            AND is_collage_sended=%s
            """
            select_id = connection.executeval(
                query_select_id, user_id, image_tarif, PENDING, 1
            )

            query_select_links = """
            SELECT image_target_link 
            FROM history_image_links 
            WHERE image_request_id=%s
            """
            select_links = connection.execute(query_select_links, select_id)

            return select_id, [
                link["image_target_link"] for link in select_links
            ]


def is_pending_collage(user_id, db):
    """
    Возвращает ссылку на колаж, который имеет статус pending
    (то есть не продан).
    """

    with db.cursor() as connection:
        with connection.transaction():
            query_select_links = """
            SELECT image_collage_link, user_images_id 
            FROM history_collage_links 
            JOIN user_images 
            ON history_collage_links.user_images_id = user_images.id 
            WHERE history_collage_links.user_id=%s 
            AND user_images.status=%s
            AND (user_images.is_collage_sended=%s OR user_images.is_collage_sended=%s)
            """
            select_links = connection.executerow(
                query_select_links, user_id, PENDING, 0, 1
            )

            return select_links if select_links else False


def is_collage_exist(user_id, collage_type, db):
    """
    Возвращает ссылку на колаж, который имеет статус pending
    (то есть не продан). Проверка перед оплатой колажа.
    """

    with db.cursor() as connection:
        with connection.transaction():
            query_select_links = """
            SELECT image_collage_link, user_images_id 
            FROM history_collage_links 
            JOIN user_images 
            ON history_collage_links.user_images_id=user_images.id 
            WHERE history_collage_links.user_id=%s 
            AND history_collage_links.collage_type=%s
            AND user_images.status=%s
            AND user_images.is_collage_sended=%s
            """
            select_links = connection.executerow(
                query_select_links, user_id, collage_type, PENDING, 1
            )

            return select_links if select_links else False


def get_image_target_links(image_request_id, db):
    """
    Возвращает массив ссылок на готовые изображения для создания коллажа.
    """

    with db.cursor() as connection:
        query_select_links = """
        SELECT image_target_link 
        FROM history_image_links 
        WHERE image_request_id=%s
        """
        connection.execute(query_select_links, (image_request_id,))
        select_links = connection.fetchone()

        return select_links[0].split(";;")


def set_user_image_links(message, path, image_count, db):
    """
    Функция добавляет ссылки на фотографии пользователя в
    таблицу user_image_links, а также обновляет количество загруженных
    фотографий в таблице user.
    """

    with db.cursor() as connection:
        with connection.transaction():
            query_insert_link = """
            INSERT INTO user_image_links (user_id, image_link) 
            VALUES (%s, %s)
            """
            connection.execute(query_insert_link, message.from_id, path)

            query_update_count = """
            UPDATE "users" SET user_images=%s WHERE user_id=%s
            """
            connection.execute(
                query_update_count, image_count, message.from_id
            )

            return True


def update_user_image_links(message, path, db):
    """
    Функция обновляет ссылку на фотографию пользователя в
    таблице user_image_links.
    """

    with db.cursor() as connection:
        with connection.transaction():
            query_update_link = """
            UPDATE user_image_links 
            SET image_link=%s
            WHERE user_id=%s
            """
            connection.execute(query_update_link, path, message.from_id)

            return True


def check_free_queue(request_type, db):
    """Returns user_id of all users in the queue."""

    with db.cursor() as connection:
        query_select_id = """
        SELECT user_id 
        FROM queue WHERE request_type=%s
        """
        user_ids = connection.execute(query_select_id, request_type)

    return [user_id[0] for user_id in user_ids]


def add_pending_image_request(user_id, image_tarif, db):
    """
    Добавляет запрос пользователя на генерацию изображений в user_images.
    """

    query_insert = """
        INSERT INTO user_images (
            user_id,
            status,
            image_tarif,
            is_generation_started, 
            date_sent
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """
    values = (
        user_id,
        PENDING,
        image_tarif,
        0,
        str(datetime.now()),
    )

    with db.cursor() as connection:
        with connection.transaction():
            record = connection.executerow(query_insert, *values)

    return record["id"]


def add_history_collage_links(user_id, collage_type, id_in_user_images, db):
    """
    Добавляет строку с новым коллажом в таблицу.
    Возвращает id новоой строки.
    """

    query_insert = """
        INSERT INTO history_collage_links (
            user_id,
            user_images_id,
            collage_type
        ) VALUES (%s, %s, %s)
        RETURNING id
    """
    values = (
        user_id,
        id_in_user_images,
        collage_type,
    )

    with db.cursor() as connection:
        connection.execute(query_insert, values)
        inserted_id = connection.fetchone()[0]
        db.commit()
    return inserted_id


def add_to_history_image_links(
    image_request_id,
    user_id,
    image_request_number,
    image_reference_links,
    image_target_links,
    current_user_image_link,
    db,
    id_in_history_collage_links=None,
):
    """
    Добавляет запрос пользователя на генерацию изображений
    в history_image_links для сохранения истории.
    """

    query_history = """
        INSERT INTO history_image_links (
            image_request_id,
            user_id,
            image_request_number,
            image_reference_link,
            image_target_link,
            user_image_link,
            collage_link_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    values_history = (
        image_request_id,
        user_id,
        image_request_number,
        image_reference_links,
        image_target_links,
        current_user_image_link,
        id_in_history_collage_links,
    )

    with db.cursor() as connection:
        connection.execute(query_history, values_history)
        db.commit()

    return True


# def update_target_image_link_in_history_image_links(
#     image_request_id, reference_path, target_image_link, db
# ):
#     """Обновляет target_image_link в таблице history_image_links."""

#     query = """
#         UPDATE history_image_links
#         SET image_target_link = %s
#         WHERE image_request_id = %s AND image_reference_link = %s
#     """

#     with db.cursor() as connection:
#         connection.execute(
#             query, target_image_link, image_request_id, reference_path
#         )

#     return True


def check_if_use_free_tarif(message, db):
    """
    Проверяет использовал ли пользователь тариф free.
    Возвращает True если использовал, либо False
    """

    query = """
        SELECT user_id FROM user_images 
        WHERE user_id=%s AND image_tarif=%s
    """

    with db.cursor() as connection:
        result = connection.execute(query, message.chat.id, "image_free")

    return len(result) > 0


def get_first_request_type_in_queue(db):
    """
    Возвращает первый запрос в очереди queue.
    Также Обновляет в queue status запроса на генерацию
    и в user_image is_generation_started = 1.
    """

    query_select = """
    SELECT * FROM queue 
    JOIN user_image_links 
    ON queue.user_id = user_image_links.user_id 
    WHERE queue.request_type=%s
    AND queue.status=%s
    FOR UPDATE SKIP LOCKED
    """
    query_image = """
    UPDATE user_images SET is_generation_started = 1 
    WHERE id = %s
    """
    query_queue = """
    UPDATE queue SET status = %s 
    WHERE id = %s AND status = %s
    """

    with db.cursor() as connection:
        for request_type in [
            IMAGE_PAID,
            IMAGE_FOR_REFERAL,
            IMAGE_COLLAGE_100,
        ]:
            connection.execute(query_select, (request_type, PENDING))
            result = connection.fetchone()
            if result:
                values = (DRAWING, result[0], PENDING)
                values_image = (result[4],)
                connection.execute(query_queue, values)
                connection.execute(query_image, values_image)
                db.commit()
                return result
    return None


def get_first_in_check_user_image_queue(db):
    """Возвращает первый запрос в очереди check_user_image_queue."""

    query = """
    SELECT * FROM check_user_image_queue 
    WHERE status=%s
    FOR UPDATE SKIP LOCKED
    """
    query_update_image_queue = """
                UPDATE check_user_image_queue 
                SET status=%s
                WHERE id = %s
                """

    with db.cursor() as connection:
        connection.execute(query, (PENDING,))
        result = connection.fetchone()
        if result:
            values_update_image_queue = (DRAWING, result[0])
            connection.execute(
                query_update_image_queue, values_update_image_queue
            )
            db.commit()
            return result

    return None


def change_find_face_status(db, id, status, image_path, check_analyse_face):
    """Обновляет статус в check_user_image_queue, распознано лицо или нет."""

    with db.cursor() as connection:
        query_update = """
                    UPDATE check_user_image_queue 
                    SET status=%s, check_analyse_face=%s, user_image_link=%s
                    WHERE id = %s
                    """
        values_update = (
            status,
            check_analyse_face,
            image_path,
            id,
        )
        connection.execute(query_update, values_update)
        db.commit()


def get_all_request_type_in_queue(db):
    """
    Возвращает все строки в очереди в соответствии с запросом.
    """

    query = """
    SELECT * FROM queue 
    JOIN user_image_links 
    ON queue.user_id = user_image_links.user_id 
    WHERE queue.request_type IN (%s, %s)
    AND queue.status=%s
    """

    with db.cursor() as connection:
        request_types = [IMAGE_PAID, IMAGE_FREE]
        results = connection.execute(query, *request_types, PENDING)

    return results


# def change_in_queue_image_request_status_to_drawing(
#     first_request_in_queue_id, image_request_id, db
# ):
#     """
#     Обновляет в queue status запроса на генерацию
#     и в user_image is_generation_started = 1.
#     """

#     query_image = """
#     UPDATE user_images SET is_generation_started = 1
#     WHERE id = %s
#     """
#     query = """
#     UPDATE queue SET status = %s
#     WHERE id = %s AND status = %s
#     """

#     with db.cursor() as connection:
#         values_image = (image_request_id,)
#         values = (DRAWING, first_request_in_queue_id, PENDING)
#         connection.execute(query, values)
#         connection.execute(query_image, values_image)
#         db.commit()


def remove_requests_from_queue(queue_id, image_request_id, db):
    """
    Удаляет строки в queue c request_type и image_request_id и
    и обновляет status user_images на READY
    """

    with db.cursor() as connection:
        query_select = """
        UPDATE user_images 
        SET status=%s
        WHERE id = %s
        """

        query_delete = """
        DELETE FROM queue 
        WHERE id = %s 
        AND image_request_id = %s
        """
        values_select = (READY, image_request_id)
        values_delete = (
            queue_id,
            image_request_id,
        )

        connection.execute(query_delete, values_delete)
        connection.execute(query_select, values_select)
        db.commit()
    return True


def update_image_colage_link(user_image_id, image_collage_link, db):
    """
    Обновляет ссылку на колаж image_collage_link в
    history_collage_links по user_images_id.
    """

    with db.cursor() as connection:
        query_update = """
            UPDATE history_collage_links SET image_collage_link = %s 
            WHERE user_images_id = %s
        """

        values = (image_collage_link, user_image_id)
        connection.execute(query_update, values)
        db.commit()
    return True


def check_referal_exists(args, new_user_id, db):
    """
    Проверяет реферальный код, и если находит в таблице user,
    возвращает этот код, иначе возвращает None.
    Также добавлят данные в таблицу referals.
    Также возвращает user_id реферера чтобы отправить ему бонус.
    """

    if args == "":
        return None, None

    with db.cursor() as connection:
        result_ref = connection.executerow(
            """SELECT * FROM "users"
            WHERE ref_code=%s
            """,
            args,
        )
        if not result_ref:
            return None, None

        result = connection.executerow(
            """SELECT * FROM "users"
            WHERE user_id=%s
            """,
            new_user_id,
        )
        if result:
            return None, None

    return args, result_ref["user_id"]
