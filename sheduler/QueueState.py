import queue


class QueueState:
    def __init__(self):
        """Ниже статусы для обеспечения конвеера из трех подзадач."""

        self.get_from_minio_task = queue.Queue()
        self.swap_task = queue.Queue()
        self.put_to_minio_task = queue.Queue()
