"""A live overlay: thin boxes and labels drawn over the running game.

Two halves, split on purpose
----------------------------
`render` is pure — entities in, a transparent RGBA image out. It touches no
window, no platform API, and nothing stateful, so it is fully testable on any
machine and the same function draws the live overlay, the replay annotations
and anything else that wants to see what perception believes.

`Win32Overlay` is the part that puts that image on screen, and it is the only
code here that cannot be verified off Windows. Keeping the drawing out of it
means a bug in the picture and a bug in the window are never the same bug.

What the labels say, and what they deliberately do not
-----------------------------------------------------
"name and certainty" needs care, because the two perception paths mean
different things by confidence and conflating them would mislead at a glance.

A trained detector reports a real per-card score, so a named box reads
`knight 0.87`. Geometry (`discover.py`) never names anything — it answers
where, what kind and whose, and its `score` is 1.0 because it either found
something or did not. Printing `knight 1.00` there would be a lie twice
over: there is no card name and there is no identity confidence.

So an unnamed entity reads `troop ?`. The box, the colour and the kind are
all things that *were* established; the question mark is the part that was
not. A glance at the overlay then tells you which of the two is running and
how far the pipeline has got, which is most of what you want an overlay for.

Colour is team and nothing else — blue for yours, red for theirs — because
that is the one property worth reading without focusing on it.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from ctypes import wintypes

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bright enough to read over grass, arena purple and spell VFX alike.
TEAM_COLORS = {
    "friendly": (72, 160, 255),
    "hostile": (255, 76, 76),
}
UNKNOWN_COLOR = (200, 200, 200)
# Text sits on a dark plate. Without it the label is unreadable the moment a
# unit walks over anything pale, which on this board is most of the time.
LABEL_BACKING = (0, 0, 0, 170)
LABEL_TEXT = (255, 255, 255, 255)


@dataclass(frozen=True)
class OverlayBox:
    """One rectangle and its caption, in captured-frame pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple[int, int, int]
    label: str


def label_for(entity, show_confidence: bool = True) -> str:
    """The caption for one entity.

    `card` empty means identity was not established — by a detector that
    abstained below its threshold, or by geometry that never attempts it. In
    both cases the kind is still known, so the label keeps what is real and
    marks the rest unknown rather than printing a number that does not refer
    to anything.
    """
    kind = getattr(entity, "kind", None)
    card = getattr(entity, "card", "") or ""
    if not card:
        # Kind can be unknown too, on a track that has not moved enough to
        # say. "? " plus a stray word would read as a card name; a bare "?"
        # reads as what it is.
        if kind is None:
            return "?"
        return f"{getattr(kind, 'value', None) or kind} ?"
    if not show_confidence:
        return card
    score = getattr(entity, "identity_score", None) or getattr(entity, "score", None)
    return f"{card} {score:.2f}" if score else card


def boxes_for(entities, show_confidence: bool = True) -> list[OverlayBox]:
    """Entities (detections or discoveries) -> boxes ready to draw."""
    boxes = []
    for entity in entities:
        team = getattr(entity, "team", "")
        boxes.append(OverlayBox(
            x0=float(entity.x0), y0=float(entity.y0),
            x1=float(entity.x1), y1=float(entity.y1),
            color=TEAM_COLORS.get(team, UNKNOWN_COLOR),
            label=label_for(entity, show_confidence)))
    return boxes


def _font(size: int):
    """A readable font if the system has one, the bitmap default otherwise.

    Never raises: an overlay that refuses to draw because a font is missing
    is worse than one with ugly text, and the font available differs across
    every machine this might run on.
    """
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, AttributeError):
            continue
    return ImageFont.load_default()


def render(
    size: tuple[int, int],
    boxes,
    line_width: int = 1,
    font_size: int = 12,
    over: Image.Image | None = None,
) -> Image.Image:
    """Draw `boxes` onto a transparent layer the size of the capture.

    `over` composites onto a copy of that frame instead, which is what the
    replay annotations want; the live overlay leaves it None so everything
    not drawn stays see-through.

    Thin by request and by sense: a 1px outline shows where a unit is without
    hiding what it looks like, which matters when the thing you are checking
    *is* whether the box is on the unit.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(font_size)

    for box in boxes:
        x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
        x1 = min(size[0] - 1, int(box.x1))
        y1 = min(size[1] - 1, int(box.y1))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=(*box.color, 255), width=line_width)
        if box.label:
            _draw_label(draw, box, (x0, y0), size, font)

    if over is None:
        return layer
    base = over.convert("RGBA")
    return Image.alpha_composite(base, layer)


def _draw_label(draw, box: OverlayBox, anchor, size, font) -> None:
    """Caption above the box, or inside its top when there is no room.

    Clamped to the frame rather than allowed to run off it: a label that
    leaves the capture is a label you cannot read, and units spend much of a
    match near the edges.
    """
    x0, y0 = anchor
    left, top, right, bottom = draw.textbbox((0, 0), box.label, font=font)
    text_w, text_h = right - left, bottom - top
    pad = 2
    plate_h = text_h + pad * 2
    plate_y = y0 - plate_h
    if plate_y < 0:                      # no room above; sit inside the box
        plate_y = y0
    plate_x = min(x0, size[0] - (text_w + pad * 2))
    plate_x = max(0, plate_x)

    draw.rectangle([plate_x, plate_y,
                    plate_x + text_w + pad * 2, plate_y + plate_h],
                   fill=LABEL_BACKING)
    draw.text((plate_x + pad - left, plate_y + pad - top), box.label,
              font=font, fill=LABEL_TEXT)


# --------------------------------------------------------------- windows


class Win32Overlay:
    """A click-through, always-on-top window pinned over the game's client area.

    Why a layered window and `UpdateLayeredWindow` rather than a coloured
    window with a transparency key: per-pixel alpha. A colour key makes one
    colour invisible, which means antialiased box edges and the label plate
    either get a hard fringe or have to be drawn without blending at all.
    `UpdateLayeredWindow` takes a 32-bit surface with real alpha and composites
    it properly, and because it swaps the whole surface at once there is no
    paint cycle to flicker.

    The extended styles each earn their place:

    * `WS_EX_TRANSPARENT` — clicks pass through to the game. Without it the
      overlay eats every tap, including the ones the bridge is trying to send.
    * `WS_EX_NOACTIVATE` and `WS_EX_TOOLWINDOW` — it never takes focus and
      never appears in the taskbar or Alt-Tab. An overlay that can steal
      focus from a game is a bug you notice mid-match.
    * `WS_EX_TOPMOST` — above the game, which is the entire point.

    **Premultiplied alpha is not optional.** `UpdateLayeredWindow` expects
    each colour channel already multiplied by its own alpha. Hand it straight
    RGBA and everything semi-transparent renders too bright with pale fringes
    around it — it looks like a colour-space bug and it is not one.

    This class is the only code in the module that cannot be tested off
    Windows. It is deliberately thin for that reason: it positions a window
    and blits an image somebody else drew.
    """

    _CLASS_NAME = "SiegeOverlayWindow"

    # Window styles
    _WS_POPUP = 0x80000000
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_TRANSPARENT = 0x00000020
    _WS_EX_TOPMOST = 0x00000008
    _WS_EX_NOACTIVATE = 0x08000000
    _WS_EX_TOOLWINDOW = 0x00000080
    _SW_SHOWNOACTIVATE = 4
    _SW_HIDE = 0
    _ULW_ALPHA = 0x00000002
    _AC_SRC_OVER = 0x00
    _AC_SRC_ALPHA = 0x01

    def __init__(self):
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("The overlay window is only available on Windows. "
                               "Use `python -m src.live.replay --annotate` to see "
                               "the same boxes on recorded frames instead.")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
        self._user32.SetProcessDPIAware()
        self._hwnd = self._create_window()
        self._visible = False

    # ------------------------------------------------------------ lifecycle

    def _create_window(self) -> int:
        user32 = self._user32
        wndproc_type = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND,
                                          ctypes.c_uint, wintypes.WPARAM,
                                          wintypes.LPARAM)
        # Held on the instance because the window class keeps a raw pointer to
        # it: if Python collects the thunk, the next message dispatched to this
        # window jumps into freed memory.
        self._wndproc = wndproc_type(
            lambda hwnd, msg, wparam, lparam:
            user32.DefWindowProcW(hwnd, msg, wparam, lparam))

        class _WNDCLASS(ctypes.Structure):
            _fields_ = [("style", ctypes.c_uint),
                        ("lpfnWndProc", wndproc_type),
                        ("cbClsExtra", ctypes.c_int),
                        ("cbWndExtra", ctypes.c_int),
                        ("hInstance", wintypes.HINSTANCE),
                        ("hIcon", wintypes.HICON),
                        ("hCursor", wintypes.HANDLE),
                        ("hbrBackground", wintypes.HBRUSH),
                        ("lpszMenuName", wintypes.LPCWSTR),
                        ("lpszClassName", wintypes.LPCWSTR)]

        cls = _WNDCLASS()
        cls.lpfnWndProc = self._wndproc
        cls.lpszClassName = self._CLASS_NAME
        cls.hInstance = None
        # A second run in the same process would otherwise fail to register.
        # Re-registration is the expected case, not an error worth raising.
        user32.RegisterClassW(ctypes.byref(cls))

        hwnd = user32.CreateWindowExW(
            self._WS_EX_LAYERED | self._WS_EX_TRANSPARENT | self._WS_EX_TOPMOST
            | self._WS_EX_NOACTIVATE | self._WS_EX_TOOLWINDOW,
            self._CLASS_NAME, "Siege overlay", self._WS_POPUP,
            0, 0, 1, 1, None, None, None, None)
        if not hwnd:
            raise RuntimeError(
                f"Could not create the overlay window "
                f"(error {ctypes.get_last_error()}).")
        return hwnd

    def close(self) -> None:
        if getattr(self, "_hwnd", None):
            self._user32.DestroyWindow(self._hwnd)
            self._hwnd = 0

    def hide(self) -> None:
        if self._visible:
            self._user32.ShowWindow(self._hwnd, self._SW_HIDE)
            self._visible = False

    # --------------------------------------------------------------- drawing

    def show(self, image: Image.Image, origin: tuple[int, int]) -> None:
        """Put `image` on screen at `origin`, in screen coordinates.

        `origin` is the capture origin the device already tracks, so the
        overlay lands exactly over the pixels perception was given. Deriving
        it any other way — from the window rect, say — would drift by
        whatever letterboxing the capture trimmed, and every box would sit a
        few pixels off in a way that looks like a homography error.
        """
        width, height = image.size
        bgra = self._premultiplied_bgra(image)

        hdc_screen = self._user32.GetDC(None)
        hdc_mem = self._gdi32.CreateCompatibleDC(hdc_screen)
        bitmap = old = None
        try:
            bitmap, pixels = self._create_dib(hdc_screen, width, height)
            ctypes.memmove(pixels, bgra.ctypes.data, bgra.nbytes)
            old = self._gdi32.SelectObject(hdc_mem, bitmap)

            if not self._visible:
                self._user32.ShowWindow(self._hwnd, self._SW_SHOWNOACTIVATE)
                self._visible = True

            size = wintypes.SIZE(width, height)
            dst = wintypes.POINT(int(origin[0]), int(origin[1]))
            src = wintypes.POINT(0, 0)
            blend = _BLENDFUNCTION(self._AC_SRC_OVER, 0, 255, self._AC_SRC_ALPHA)
            self._user32.UpdateLayeredWindow(
                self._hwnd, hdc_screen, ctypes.byref(dst), ctypes.byref(size),
                hdc_mem, ctypes.byref(src), 0, ctypes.byref(blend),
                self._ULW_ALPHA)
        finally:
            if old:
                self._gdi32.SelectObject(hdc_mem, old)
            if bitmap:
                self._gdi32.DeleteObject(bitmap)
            self._gdi32.DeleteDC(hdc_mem)
            self._user32.ReleaseDC(None, hdc_screen)
        self.pump()

    def _create_dib(self, hdc, width: int, height: int):
        header = _BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        header.biWidth = width
        # Negative height for a top-down DIB, matching the row order PIL and
        # numpy use. Positive would render the overlay upside down, which is
        # an entertaining way to lose an afternoon.
        header.biHeight = -height
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0            # BI_RGB
        pixels = ctypes.c_void_p()
        bitmap = self._gdi32.CreateDIBSection(
            hdc, ctypes.byref(header), 0, ctypes.byref(pixels), None, 0)
        if not bitmap:
            raise RuntimeError("Could not allocate the overlay bitmap.")
        return bitmap, pixels

    def pump(self) -> None:
        """Drain this window's message queue.

        A window that never pumps is eventually declared unresponsive by the
        shell, which greys it out. Nothing here needs the messages — the
        wndproc is `DefWindowProc` — they just have to be taken.
        """
        msg = wintypes.MSG()
        while self._user32.PeekMessageW(ctypes.byref(msg), self._hwnd, 0, 0, 1):
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

    @staticmethod
    def _premultiplied_bgra(image: Image.Image) -> np.ndarray:
        """RGBA -> the premultiplied BGRA `UpdateLayeredWindow` requires.

        Both parts are mandatory and both fail quietly. Wrong channel order
        swaps red and blue, so the team colours invert — which looks like a
        logic bug in the overlay rather than a byte-order one. Skipping the
        premultiply makes every blended pixel too bright with pale fringes.
        """
        rgba = np.asarray(image.convert("RGBA"), np.uint16)
        alpha = rgba[:, :, 3:4]
        rgb = (rgba[:, :, :3] * alpha // 255).astype(np.uint8)
        return np.dstack([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0],
                          alpha[:, :, 0].astype(np.uint8)]).copy()


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_byte),
                ("AlphaFormat", ctypes.c_byte)]
