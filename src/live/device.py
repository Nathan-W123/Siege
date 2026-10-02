"""Screen and input transports for calibrated live-play sessions."""
from __future__ import annotations

import ctypes
import subprocess
import time
from ctypes import wintypes
from io import BytesIO
from typing import Protocol

import numpy as np
from PIL import Image, ImageGrab

PT_TOUCH = 0x00000002
POINTER_FLAG_INRANGE = 0x00000002
POINTER_FLAG_INCONTACT = 0x00000004
POINTER_FLAG_DOWN = 0x00010000
POINTER_FLAG_UP = 0x00040000
TOUCH_MASK_CONTACTAREA = 0x00000001
TOUCH_MASK_ORIENTATION = 0x00000002
TOUCH_MASK_PRESSURE = 0x00000004
TOUCH_FEEDBACK_DEFAULT = 0x00000001


class _PointerInfo(ctypes.Structure):
    _fields_ = [
        ("pointer_type", ctypes.c_int32),
        ("pointer_id", ctypes.c_uint32),
        ("frame_id", ctypes.c_uint32),
        ("pointer_flags", ctypes.c_int32),
        ("source_device", wintypes.HANDLE),
        ("hwnd_target", wintypes.HWND),
        ("pt_pixel_location", wintypes.POINT),
        ("pt_himetric_location", wintypes.POINT),
        ("pt_pixel_location_raw", wintypes.POINT),
        ("pt_himetric_location_raw", wintypes.POINT),
        ("dw_time", wintypes.DWORD),
        ("history_count", ctypes.c_uint32),
        ("input_data", ctypes.c_int32),
        ("dw_key_states", wintypes.DWORD),
        ("performance_count", ctypes.c_uint64),
        ("button_change_type", ctypes.c_int32),
    ]


class _PointerTouchInfo(ctypes.Structure):
    _fields_ = [
        ("pointer_info", _PointerInfo),
        ("touch_flags", ctypes.c_int32),
        ("touch_mask", ctypes.c_int32),
        ("rc_contact", wintypes.RECT),
        ("rc_contact_raw", wintypes.RECT),
        ("orientation", ctypes.c_uint32),
        ("pressure", ctypes.c_uint32),
    ]


def _win_error(message: str) -> RuntimeError:
    code = ctypes.get_last_error()
    return RuntimeError(f"{message}: {ctypes.FormatError(code).strip()} (WinError {code})")


class LiveDevice(Protocol):
    """Minimal interface required by :class:`LiveMatchRunner`."""

    def screenshot(self) -> Image.Image: ...

    def tap(self, x: int, y: int) -> None: ...


class ADBDevice:
    def __init__(self, adb_path: str, serial: str | None = None):
        self.adb_path = adb_path
        self.serial = serial

    def screenshot(self) -> Image.Image:
        result = self._run("exec-out", "screencap", "-p", capture_output=True)
        return Image.open(BytesIO(result.stdout)).convert("RGB")

    def tap(self, x: int, y: int) -> None:
        self._run("shell", "input", "tap", str(x), str(y))

    def _run(self, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[bytes]:
        command = [self.adb_path]
        if self.serial:
            command.extend(("-s", self.serial))
        command.extend(args)
        try:
            return subprocess.run(command, check=True, capture_output=capture_output)
        except FileNotFoundError as error:
            raise RuntimeError(f"ADB executable was not found: {self.adb_path}") from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode(errors="replace") if error.stderr else str(error)
            raise RuntimeError(f"ADB command failed: {detail}") from error


class WindowsDesktopDevice:
    """Capture and click the Windows desktop or one client window.

    In window mode, screenshots and taps use coordinates local to the game's
    client area, so the game can be moved or resized without reconfiguring its
    desktop position.  The configured coordinates are still scaled from the
    reference size by the runner.
    """

    def __init__(self, capture_mode: str = "virtual_desktop", window_title: str | None = None) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Windows desktop transport is only available on Windows.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32.InitializeTouchInjection.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        self._user32.InitializeTouchInjection.restype = wintypes.BOOL
        self._user32.InjectTouchInput.argtypes = [ctypes.c_uint32, ctypes.POINTER(_PointerTouchInfo)]
        self._user32.InjectTouchInput.restype = wintypes.BOOL
        # Keep Windows' DPI scaling from making capture and input coordinates
        # disagree on high-DPI displays.
        self._user32.SetProcessDPIAware()
        self._origin_x = self._user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        self._origin_y = self._user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
        self.capture_mode = capture_mode
        self.window_title = window_title.lower() if window_title else None
        self._capture_origin = (self._origin_x, self._origin_y)
        self._cached_bounds: tuple[int, int, int, int] | None = None
        self._cached_viewport: tuple[int, int, int, int] | None = None
        self._touch_injection_ready = False

    @property
    def capture_origin(self) -> tuple[int, int]:
        """Screen position of the captured frame's top-left pixel.

        Updated by `screenshot`, so it describes the most recent capture.
        The overlay pins itself here rather than to the window rect: the
        capture is the window's client area *minus whatever letterboxing was
        trimmed*, and using the window rect instead would offset every box by
        the trim in a way that looks like a homography error.
        """
        return self._capture_origin

    def screenshot(self) -> Image.Image:
        try:
            if self.capture_mode == "virtual_desktop":
                self._capture_origin = (self._origin_x, self._origin_y)
                return ImageGrab.grab(all_screens=True).convert("RGB")
            bounds = self._client_bounds(self._game_window())
            left, top, right, bottom = bounds
            image = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB")
            # Re-detecting black side bars every frame lets per-frame noise
            # (HUD animation near the top edge) jitter the crop width, which
            # then jitters every scaled tap coordinate. Only redo it when the
            # window has actually moved or resized.
            if bounds != self._cached_bounds:
                self._cached_viewport = self._content_viewport(image)
                self._cached_bounds = bounds
            viewport = self._cached_viewport
            self._capture_origin = (left + viewport[0], top + viewport[1])
            return image.crop(viewport)
        except OSError as error:
            raise RuntimeError("Unable to capture the Windows desktop.") from error

    def tap(self, x: int, y: int) -> None:
        if self.capture_mode != "window":
            self._tap_mouse(x, y)
            return
        # Windows silently drops synthetic clicks aimed at a window that
        # isn't foreground/focused; a background script's cursor can
        # visibly move and "click" without the game ever seeing it.
        self._focus_window(self._game_window())
        self._tap_touch(x, y)

    def _tap_mouse(self, x: int, y: int) -> None:
        origin_x, origin_y = self._capture_origin
        if not self._user32.SetCursorPos(x + origin_x, y + origin_y):
            raise _win_error(f"Unable to move the cursor to ({x}, {y})")
        self._user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
        self._user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP

    def _tap_touch(self, x: int, y: int) -> None:
        """Inject a real touch contact instead of a mouse click.

        Windows Subsystem for Android translates real hardware mouse/touch
        input to Android touch events itself, but does not reliably treat a
        synthetic `mouse_event` click as that hardware input. Touch injection
        goes through the same pointer pipeline an actual touchscreen uses,
        which WSA does recognize.
        """
        if not self._touch_injection_ready:
            if not self._user32.InitializeTouchInjection(1, TOUCH_FEEDBACK_DEFAULT):
                raise _win_error("Unable to initialize touch injection")
            self._touch_injection_ready = True

        origin_x, origin_y = self._capture_origin
        screen_x, screen_y = x + origin_x, y + origin_y

        contact = _PointerTouchInfo()
        ctypes.memset(ctypes.byref(contact), 0, ctypes.sizeof(contact))
        contact.pointer_info.pointer_type = PT_TOUCH
        contact.pointer_info.pointer_id = 0
        contact.pointer_info.pt_pixel_location.x = screen_x
        contact.pointer_info.pt_pixel_location.y = screen_y
        contact.touch_mask = TOUCH_MASK_CONTACTAREA | TOUCH_MASK_ORIENTATION | TOUCH_MASK_PRESSURE
        contact.orientation = 90
        contact.pressure = 32000
        contact.rc_contact.left = screen_x - 5
        contact.rc_contact.right = screen_x + 5
        contact.rc_contact.top = screen_y - 5
        contact.rc_contact.bottom = screen_y + 5

        contact.pointer_info.pointer_flags = POINTER_FLAG_DOWN | POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT
        if not self._user32.InjectTouchInput(1, ctypes.byref(contact)):
            raise _win_error(f"Unable to inject touch-down at ({x}, {y})")
        time.sleep(0.05)
        contact.pointer_info.pointer_flags = POINTER_FLAG_UP
        if not self._user32.InjectTouchInput(1, ctypes.byref(contact)):
            raise _win_error(f"Unable to inject touch-up at ({x}, {y})")

    def _focus_window(self, hwnd: int) -> None:
        if self._user32.GetForegroundWindow() == hwnd:
            return
        current_thread = self._kernel32.GetCurrentThreadId()
        target_thread = self._user32.GetWindowThreadProcessId(hwnd, None)
        attached = bool(self._user32.AttachThreadInput(current_thread, target_thread, True))
        try:
            self._user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                self._user32.AttachThreadInput(current_thread, target_thread, False)
        # Give the game a moment to actually regain focus before we click it.
        time.sleep(0.1)

    def _game_window(self) -> int:
        if not self.window_title:
            hwnd = self._user32.GetForegroundWindow()
            if not hwnd:
                raise RuntimeError("No foreground window is available for desktop_capture: window.")
            return hwnd

        matches: list[int] = []
        enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def find_match(hwnd, _lparam):
            if not self._user32.IsWindowVisible(hwnd):
                return True
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                self._user32.GetWindowTextW(hwnd, title, len(title))
                if self.window_title in title.value.lower():
                    matches.append(hwnd)
                    return False
            return True

        self._user32.EnumWindows(enum_proc_type(find_match), 0)
        if not matches:
            raise RuntimeError(f"No visible window title contains {self.window_title!r}.")
        return matches[0]

    def _client_bounds(self, hwnd: int) -> tuple[int, int, int, int]:
        rect = wintypes.RECT()
        if not self._user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("Unable to read the game window's client area.")
        point = wintypes.POINT(0, 0)
        if not self._user32.ClientToScreen(hwnd, ctypes.byref(point)):
            raise RuntimeError("Unable to locate the game window's client area.")
        if rect.right <= 0 or rect.bottom <= 0:
            raise RuntimeError("The game window's client area is empty.")
        return point.x, point.y, point.x + rect.right, point.y + rect.bottom

    @staticmethod
    def _content_viewport(image: Image.Image) -> tuple[int, int, int, int]:
        """Trim solid-black side bars that some Android desktop clients add.

        The game itself stays in a portrait viewport, but its host window can
        be wider. Sampling a short grayscale version keeps this cheap; the
        column scan is vectorized because the pure-Python version walked
        width x sample_height pixels one at a time (~66k reads at typical
        sizes) on a path that runs on every window move.
        """
        width, height = image.size
        sample_height = min(120, height)
        gray = np.asarray(image.convert("L").resize((width, sample_height)))
        # A column belongs to the content area if most of its sampled pixels
        # are brighter than the letterbox bars.
        active = (gray > 35).mean(axis=0) >= 0.65
        if not active.any():
            return (0, 0, width, height)
        # Longest run of consecutive active columns: boundaries are where
        # `active` flips, so diff the padded signal and pair up the edges.
        padded = np.concatenate(([False], active, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        widest = int(np.argmax(ends - starts))
        left, right = int(starts[widest]), int(ends[widest])
        # Do not turn a transient dark screen into an invalid tiny image.
        if right - left < width * 0.45:
            return (0, 0, width, height)
        return (left, 0, right, height)
