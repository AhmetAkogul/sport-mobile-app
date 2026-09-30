"""OpenCV kaynak çağrılarını süre sınırlı ayrı süreçte çalıştırır."""

import math
import multiprocessing as mp
import pickle
import queue
import threading


def _worker(connection, source, factory):
    handle = None
    try:
        if factory is None:
            import cv2
            factory = cv2.VideoCapture
        handle = factory(source)
        connection.send((True, handle.isOpened()))
        while True:
            command = connection.recv()
            if command == "close":
                # Referans **once** dusurulur: release() hata verirse finally
                # blogu ayni nesneyi ikinci kez kapatmaz (dis inceleme C.2.1).
                local = handle
                handle = None
                local.release()
                connection.send((True, None))
                break
            if isinstance(command, tuple) and command[0] in {"get", "set"}:
                method, *args = command
                connection.send((True, getattr(handle, method)(*args)))
                continue
            if command not in {"grab", "retrieve"}:
                raise ValueError("Bilinmeyen kaynak komutu")
            connection.send((True, getattr(handle, command)()))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if handle is not None:
            handle.release()
        connection.close()


def _receiver(connection, messages):
    # Büyük görüntünün yarım IPC aktarımı ana iş parçacığını kilitlemesin.
    try:
        while True:
            messages.put(connection.recv())
    except (EOFError, OSError) as exc:
        messages.put((False, f"Kaynak süreci kapandı: {type(exc).__name__}"))


class ProcessCapture:
    """Tek kaynağın open/grab/retrieve/release çağrılarına zaman aşımı uygular.

    spawn kullanır; factory verilirse modül seviyesinde, pickle edilebilir olmalı.
    Kaynak nesnesi yalnızca alt süreçte oluşturulur ve kullanılır.
    """

    def __init__(self, source, *, timeout_s=5.0, open_timeout_s=15.0, factory=None):
        for value in (timeout_s, open_timeout_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Zaman aşımı sonlu pozitif saniye olmalı.")
        if factory is not None:
            # spawn factory'yi pickle eder; lambda gibi pickle edilemeyen bir
            # nesne .start() icinde anlasilmaz bir hataya donusur. Hata erken
            # verilsin (dis inceleme C.2.2).
            try:
                pickle.dumps(factory)
            except Exception as exc:
                raise ValueError(
                    "factory modul seviyesinde ve pickle edilebilir olmali "
                    f"(lambda sozlesme disi): {type(exc).__name__}: {exc}") from exc
        self.timeout_s = timeout_s
        self._closed = False
        self._opened = False
        self._messages = queue.Queue()
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, source, factory), daemon=True)
        self._thread = threading.Thread(target=_receiver, args=(self._connection, self._messages), daemon=True)
        try:
            self._process.start()
            child.close()
            self._thread.start()
            self._opened = bool(self._receive("open", open_timeout_s))
            if not self._opened:
                self.release()
        except BaseException:
            child.close()
            self._stop()
            raise

    def _receive(self, operation, timeout):
        try:
            ok, result = self._messages.get(timeout=timeout)
        except queue.Empty:
            self._stop()
            raise TimeoutError(f"Kaynak {operation} işlemi {timeout:g} saniyede tamamlanmadı.") from None
        if not ok:
            self._stop()
            raise RuntimeError(result)
        return result

    def _request(self, command):
        if self._closed:
            raise RuntimeError("Kaynak kapalı.")
        try:
            self._connection.send(command)
            return self._receive(command, self.timeout_s)
        except BaseException:
            self._stop()
            raise

    def isOpened(self):
        return self._opened and not self._closed

    def grab(self):
        return self._request("grab")

    def retrieve(self):
        return self._request("retrieve")

    def get(self, property_id):
        return self._request(("get", property_id))

    def set(self, property_id, value):
        return self._request(("set", property_id, value))

    def release(self):
        if self._closed:
            return
        try:
            self._request("close")
        finally:
            self._stop()

    def _stop(self):
        if self._closed:
            return
        self._closed = True
        if self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=0.5)
        self._connection.close()
        if self._thread.ident is not None:
            self._thread.join(timeout=0.5)
