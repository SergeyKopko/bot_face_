import threading
from datetime import datetime
import datetime as dt
import requests
from time import sleep


def task(task_id):
    sleep(5)
    now = dt.datetime.now()
    now_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(now_formatted)
    response = requests.get("https://python.org")
    response_html = response.text

    print(f"Задача {task_id} выполнена.")


def sync_execute():
    tasks = []
    # Чтобы не создавать десять потоков отдельно,
    # запускается цикл, и потоки создаются в нём.
    for i in range(1, 11):
        # Каждый запрос к сайту должен выполняться в отдельном потоке.
        tasks.append(threading.Thread(target=task, args=(i,)))
    # Запуск дочерних потоков в цикле.
    for t in tasks:
        t.start()

    # Пока не выполнятся все дочерние потоки,
    # программа не начнёт выполнять основной поток.
    for t in tasks:
        t.join()


# Основной поток программы.
if __name__ == "__main__":
    print("Многопоточное выполнение кода:")
    start_time = datetime.now()
    sync_execute()
    end_time = datetime.now()
    print(f"Итоговое время выполнения: {end_time - start_time} секунд.")
