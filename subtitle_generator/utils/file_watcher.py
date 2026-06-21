import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    FileSystemEventHandler = object


class FileWatcher:
    POLL_INTERVAL = 2.0

    def __init__(
        self,
        directory: str | os.PathLike,
        callback: Callable[[str], None],
        supported_extensions: Optional[Set[str]] = None,
        stable_check: bool = True,
        stable_size_attempts: int = 3
    ):
        self.directory = Path(directory)
        self.callback = callback
        self.supported_extensions = supported_extensions
        self.stable_check = stable_check
        self.stable_size_attempts = stable_size_attempts
        self._processed_files: Set[str] = set()
        self._observer = None
        self._stop_event = threading.Event()

    def _is_supported(self, path: str | os.PathLike) -> bool:
        if self.supported_extensions is None:
            return True
        ext = Path(path).suffix.lower()
        return ext in self.supported_extensions

    def _wait_file_stable(self, file_path: str | os.PathLike, timeout: int = 60) -> bool:
        if not self.stable_check:
            return True

        file_path = Path(file_path)
        last_size = -1
        stable_count = 0
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                current_size = file_path.stat().st_size
            except FileNotFoundError:
                return False

            if current_size == last_size and current_size > 0:
                stable_count += 1
                if stable_count >= self.stable_size_attempts:
                    return True
            else:
                stable_count = 0
                last_size = current_size

            time.sleep(self.POLL_INTERVAL)

        return False

    def _process_file(self, file_path: str | os.PathLike):
        file_path = str(Path(file_path).resolve())
        if file_path in self._processed_files:
            return

        if not self._is_supported(file_path):
            return

        if not self._wait_file_stable(file_path):
            return

        self._processed_files.add(file_path)
        try:
            self.callback(file_path)
        except Exception:
            pass

    def scan_existing(self):
        if not self.directory.exists():
            return

        for root, _, files in os.walk(self.directory):
            for filename in files:
                if self._stop_event.is_set():
                    return
                file_path = Path(root) / filename
                self._process_file(file_path)

    def start(self, scan_existing: bool = True):
        if scan_existing:
            self.scan_existing()

        if HAS_WATCHDOG:
            self._start_watchdog()
        else:
            self._start_polling()

    def _start_watchdog(self):
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher

            def on_created(self, event):
                if not event.is_directory:
                    self.watcher._process_file(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    self.watcher._process_file(event.dest_path)

        self._observer = Observer()
        self._observer.schedule(Handler(self), str(self.directory), recursive=True)
        self._observer.start()

    def _start_polling(self):
        def poll_loop():
            while not self._stop_event.is_set():
                self.scan_existing()
                self._stop_event.wait(self.POLL_INTERVAL)

        thread = threading.Thread(target=poll_loop, daemon=True)
        thread.start()

    def stop(self):
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    def wait(self):
        try:
            while not self._stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.stop()
