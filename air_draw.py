import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
CAMERA_ID = 0
CAM_W     = 1280
CAM_H     = 720
CAM_FPS   = 60
SMOOTH_N  = 6
DET_CONF  = 0.80
TRK_CONF  = 0.80

COLORS = [
    ("Magenta", (255,   0, 255)),
    ("Cyan",    (255, 255,   0)),
    ("Green",   (  0, 255,   0)),
    ("Orange",  (  0, 165, 255)),
    ("Red",     ( 50,  50, 255)),
    ("White",   (255, 255, 255)),
    ("Blue",    (255,  80,   0)),
]
SIZES = [3, 6, 10, 16]


# ══════════════════════════════════════════════════════
#  SHAPE DETECTOR  (rewritten for reliability)
# ══════════════════════════════════════════════════════
class ShapeDetector:

    @staticmethod
    def detect(pts):
        if len(pts) < 15:
            return "free", None

        arr = np.array(pts, dtype=np.float32)

        # ── 1. LINE CHECK (highest priority if very straight) ──
        line_score = ShapeDetector._line_score(arr)
        if line_score > 0.90:
            return "line", (tuple(map(int, arr[0])), tuple(map(int, arr[-1])))

        # ── 2. CIRCLE CHECK ──
        circ_score = ShapeDetector._circle_score(arr)
        if circ_score > 0.75:
            return "circle", ShapeDetector._fit_circle(arr)

        # ── 3. RECTANGLE CHECK ──
        rect_score = ShapeDetector._rect_score(arr)
        if rect_score > 0.72:
            return "rectangle", ShapeDetector._fit_rect(arr)

        # ── fallback: line with lower threshold ──
        if line_score > 0.80:
            return "line", (tuple(map(int, arr[0])), tuple(map(int, arr[-1])))

        return "free", None

    # ── LINE ─────────────────────────────────────────
    @staticmethod
    def _line_score(p):
        if len(p) < 2:
            return 0.0
        try:
            vx, vy, cx, cy = cv2.fitLine(
                p.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            perp = np.abs((p[:, 0]-cx)*(-vy) + (p[:, 1]-cy)*vx)
            span = (p[:, 0]-cx)*vx + (p[:, 1]-cy)*vy
            length = span.max() - span.min()
            if length < 30:
                return 0.0
            straightness = 1.0 - (perp.mean() / (length + 1e-6))
            return float(np.clip(straightness, 0, 1))
        except Exception:
            return 0.0

    # ── CIRCLE ───────────────────────────────────────
    @staticmethod
    def _circle_score(p):
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        dists  = np.sqrt((p[:, 0]-cx)**2 + (p[:, 1]-cy)**2)
        r      = dists.mean()
        if r < 15:
            return 0.0
        # how uniform are the radii?
        uniformity = 1.0 - (dists.std() / (r + 1e-6))
        # does the stroke close back to start?
        closure_dist = np.linalg.norm(p[0] - p[-1])
        closure = max(0.0, 1.0 - closure_dist / (2 * r + 1e-6))
        # does it span 270+ degrees? (checks angular coverage)
        angles = np.arctan2(p[:, 1]-cy, p[:, 0]-cx)
        angle_range = np.ptp(np.unwrap(angles)) / (2 * np.pi)
        coverage = min(1.0, angle_range / 0.75)
        score = uniformity * 0.5 + closure * 0.3 + coverage * 0.2
        return float(np.clip(score, 0, 1))

    @staticmethod
    def _fit_circle(p):
        cx = int(p[:, 0].mean())
        cy = int(p[:, 1].mean())
        r  = int(np.sqrt((p[:, 0]-cx)**2 + (p[:, 1]-cy)**2).mean())
        return cx, cy, max(r, 5)

    # ── RECTANGLE ────────────────────────────────────
    @staticmethod
    def _rect_score(p):
        try:
            hull  = cv2.convexHull(p.astype(np.int32))
            hull_area = cv2.contourArea(hull)
            if hull_area < 500:
                return 0.0
            rect  = cv2.minAreaRect(p)
            box   = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            if box_area < 500:
                return 0.0
            # how well does convex hull fill the bounding rect?
            fill = hull_area / (box_area + 1e-6)
            # does stroke close?
            diag = np.sqrt(box_area)
            closure = max(0.0, 1.0 - np.linalg.norm(p[0]-p[-1]) / (diag * 0.5 + 1e-6))
            # aspect ratio check (not a tiny sliver)
            w_r, h_r = rect[1]
            aspect = min(w_r, h_r) / (max(w_r, h_r) + 1e-6)
            aspect_ok = 1.0 if aspect > 0.15 else aspect / 0.15
            score = fill * 0.5 + closure * 0.3 + aspect_ok * 0.2
            return float(np.clip(score, 0, 1))
        except Exception:
            return 0.0

    @staticmethod
    def _fit_rect(p):
        box = cv2.boxPoints(cv2.minAreaRect(p))
        return box.astype(int)


# ══════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════
class AirDraw:

    BAR_H    = 58
    SWATCH_W = 76

    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        mp_h = mp.solutions.hands
        self.hands = mp_h.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=DET_CONF,
            min_tracking_confidence=TRK_CONF,
        )
        self.HAND_CONN = mp_h.HAND_CONNECTIONS
        self.mp_draw   = mp.solutions.drawing_utils
        self.d_dot  = self.mp_draw.DrawingSpec(color=(0,255,200), thickness=-1, circle_radius=4)
        self.d_line = self.mp_draw.DrawingSpec(color=(80,80,255), thickness=1)

        self.canvas      = None
        self.shape_layer = None

        self.color_idx = 0
        self.color     = COLORS[0][1]
        self.size_idx  = 1
        self.size      = SIZES[1]

        self.smooth_buf = deque(maxlen=SMOOTH_N)
        self.stroke_pts = []
        self.history    = []
        self.prev_x     = 0
        self.prev_y     = 0

        self.snap        = True
        self.flash_label = ""
        self.flash_score = 0.0
        self.flash_t     = 0

        self.bar_vis    = False
        self.bar_timer  = 0
        self.bar_bounds = None

        self.show_help = False
        self.fps_buf   = deque(maxlen=30)
        self.last_t    = time.perf_counter()

        self.frame = None
        self.lock  = threading.Lock()
        self.alive = True
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while self.alive:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def _smooth(self, x, y):
        self.smooth_buf.append((x, y))
        return (int(np.mean([p[0] for p in self.smooth_buf])),
                int(np.mean([p[1] for p in self.smooth_buf])))

    # ── commit stroke → shape snap ───────────────────
    def _commit(self):
        pts = self.stroke_pts
        if len(pts) < 5:
            self.stroke_pts = []
            return

        shape, params = ShapeDetector.detect(pts)
        self.flash_label = shape
        self.flash_t     = time.perf_counter()
        c, th = self.color, self.size

        if self.snap and shape != "free" and params is not None:
            # clear rough stroke region
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            pad = 30
            hh, ww = self.canvas.shape[:2]
            x1 = max(0, min(xs)-pad);  y1 = max(0, min(ys)-pad)
            x2 = min(ww, max(xs)+pad); y2 = min(hh, max(ys)+pad)
            self.canvas[y1:y2, x1:x2]      = 0
            self.shape_layer[y1:y2, x1:x2] = 0

            if shape == "circle":
                cx, cy, r = params
                cv2.circle(self.shape_layer, (cx, cy), r, c, th)

            elif shape == "rectangle":
                box = params
                for i in range(4):
                    cv2.line(self.shape_layer,
                             tuple(box[i]), tuple(box[(i+1) % 4]), c, th)

            elif shape == "line":
                p1, p2 = params
                cv2.line(self.shape_layer, p1, p2, c, th)

        self.history.append({
            "pts": pts[:], "shape": shape,
            "params": params, "color": c, "size": th,
        })
        self.stroke_pts = []

    # ── undo ─────────────────────────────────────────
    def _undo(self):
        if not self.history:
            return
        self.history.pop()
        self.canvas[:]      = 0
        self.shape_layer[:] = 0
        for s in self.history:
            c, th = s["color"], s["size"]
            if self.snap and s["shape"] != "free" and s["params"] is not None:
                if s["shape"] == "circle":
                    cx, cy, r = s["params"]
                    cv2.circle(self.shape_layer, (cx, cy), r, c, th)
                elif s["shape"] == "rectangle":
                    for i in range(4):
                        cv2.line(self.shape_layer,
                                 tuple(s["params"][i]),
                                 tuple(s["params"][(i+1)%4]), c, th)
                elif s["shape"] == "line":
                    cv2.line(self.shape_layer, s["params"][0], s["params"][1], c, th)
            else:
                for i in range(1, len(s["pts"])):
                    cv2.line(self.canvas, s["pts"][i-1], s["pts"][i], c, th)

    # ── colour bar ───────────────────────────────────
    def _draw_bar(self, img):
        h, w    = img.shape[:2]
        total_w = len(COLORS) * self.SWATCH_W
        bx      = (w - total_w) // 2
        by      = 14
        cv2.rectangle(img, (bx-10, by-10),
                      (bx+total_w+10, by+self.BAR_H+22), (15,15,15), -1)
        cv2.rectangle(img, (bx-10, by-10),
                      (bx+total_w+10, by+self.BAR_H+22), (120,120,120), 1)
        for i, (name, val) in enumerate(COLORS):
            sx = bx + i*self.SWATCH_W
            cv2.rectangle(img, (sx+2, by), (sx+self.SWATCH_W-2, by+self.BAR_H), val, -1)
            if i == self.color_idx:
                cv2.rectangle(img, (sx, by-3),
                              (sx+self.SWATCH_W, by+self.BAR_H+3), (255,255,255), 3)
            cv2.putText(img, name, (sx+4, by+self.BAR_H+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1, cv2.LINE_AA)
        self.bar_bounds = (bx, by, total_w)

    def _hit_bar(self, x, y):
        if not self.bar_bounds:
            return -1
        bx, by, bw = self.bar_bounds
        if by <= y <= by+self.BAR_H and bx <= x <= bx+bw:
            return max(0, min((x-bx)//self.SWATCH_W, len(COLORS)-1))
        return -1

    # ── HUD ──────────────────────────────────────────
    def _hud(self, img, fps, hand):
        h, w = img.shape[:2]
        ov = img.copy()
        cv2.rectangle(ov, (0, h-48), (w, h), (10,10,10), -1)
        cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)

        fc = (0,255,100) if fps>=25 else (0,165,255) if fps>=15 else (0,50,255)
        cv2.putText(img, f"FPS {fps:.0f}", (10, h-16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, fc, 2, cv2.LINE_AA)
        cv2.circle(img, (118, h-24), 9, (0,255,100) if hand else (60,60,60), -1)
        cv2.rectangle(img, (140, h-42), (184, h-6), self.color, -1)
        cv2.rectangle(img, (138, h-44), (186, h-4), (255,255,255), 1)
        cv2.circle(img, (212, h-24), self.size, (210,210,210), -1)

        snap_c = (0,255,100) if self.snap else (80,80,80)
        cv2.putText(img, "SNAP " + ("ON" if self.snap else "OFF"),
                    (238, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, snap_c, 1, cv2.LINE_AA)

        # shape flash (centre screen)
        if time.perf_counter()-self.flash_t < 1.5 and self.flash_label:
            icons = {"circle": "O  CIRCLE", "rectangle": "[] RECTANGLE",
                     "line": "-- LINE", "free": "~  FREEHAND"}
            label = icons.get(self.flash_label, self.flash_label.upper())
            col   = (0,255,180) if self.flash_label != "free" else (160,160,160)
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
            cv2.putText(img, label, ((w-tw)//2, h-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)

        hint = "  ☝Draw  ✌Erase  🤙ColourBar  🖐Lift  [S]Snap  [Z]Undo  [C]Clear  [+/-]Size  [H]Help  [ESC]Quit"
        cv2.putText(img, hint, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130,130,130), 1, cv2.LINE_AA)

        if self.show_help:
            pw, ph = 560, 380
            px, py = (w-pw)//2, (h-ph)//2
            ov2 = img.copy()
            cv2.rectangle(ov2, (px,py), (px+pw,py+ph), (12,12,12), -1)
            cv2.addWeighted(ov2, 0.90, img, 0.10, 0, img)
            cv2.rectangle(img, (px,py), (px+pw,py+ph), (200,200,200), 1)
            rows = [
                ("AIR DRAW  -  HELP", True),
                ("", False),
                ("GESTURES", True),
                ("  Index finger UP only      ->  Draw", False),
                ("  Index + Middle UP         ->  Erase circle", False),
                ("  Index + Pinky UP          ->  Colour bar (hover to pick)", False),
                ("  Any other pose            ->  Lift pen", False),
                ("", False),
                ("SHAPE SNAP  (draw rough, lift pen = auto corrects)", True),
                ("  Circle   : draw a loop and close it back to start", False),
                ("  Rectangle: draw a box shape and close the corner", False),
                ("  Line     : draw one straight stroke", False),
                ("  S key    : toggle shape snap ON / OFF", False),
                ("", False),
                ("KEYBOARD", True),
                ("  Z  Undo    C  Clear    +/-  Brush size    ESC  Quit", False),
            ]
            for j, (txt, bold) in enumerate(rows):
                cv2.putText(img, txt, (px+16, py+28+j*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255,255,150) if bold else (210,210,210),
                            2 if bold else 1, cv2.LINE_AA)

    # ── main loop ────────────────────────────────────
    def run(self):
        cv2.namedWindow("Air Draw", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Air Draw", CAM_W, CAM_H)

        bar_active = False

        while True:
            with self.lock:
                if self.frame is None:
                    continue
                img = self.frame.copy()

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            if self.canvas is None:
                self.canvas      = np.zeros((h, w, 3), np.uint8)
                self.shape_layer = np.zeros((h, w, 3), np.uint8)

            now = time.perf_counter()
            self.fps_buf.append(1.0 / max(now-self.last_t, 1e-6))
            self.last_t = now
            fps = float(np.mean(self.fps_buf))

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = self.hands.process(rgb)

            hand = False
            bar_active = False

            if res.multi_hand_landmarks:
                hand = True
                lms  = res.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    img, lms, self.HAND_CONN, self.d_dot, self.d_line)

                def lm(i):  return lms.landmark[i]
                def up(i):  return lm(i).y < lm(i-2).y

                i_up = up(8)
                m_up = up(12)
                r_up = up(16)
                p_up = up(20)

                rx = int(lm(8).x * w)
                ry = int(lm(8).y * h)
                sx, sy = self._smooth(rx, ry)

                cv2.circle(img, (sx, sy), self.size+6, self.color, 2)
                cv2.circle(img, (sx, sy), 3, (255,255,255), -1)

                # 🤙 Index + Pinky -> colour bar
                if i_up and p_up and not m_up and not r_up:
                    bar_active     = True
                    self.bar_vis   = True
                    self.bar_timer = time.perf_counter()
                    self._draw_bar(img)
                    hit = self._hit_bar(sx, sy)
                    if hit >= 0:
                        self.color_idx = hit
                        self.color     = COLORS[hit][1]
                    if self.stroke_pts:
                        self._commit()
                    self.prev_x = self.prev_y = 0

                # ✌ Index + Middle -> erase
                elif i_up and m_up and not r_up and not p_up:
                    er = 32
                    mx = int(lm(12).x * w)
                    my = int(lm(12).y * h)
                    cv2.circle(img,              (mx, my), er, (80,80,80), 2)
                    cv2.circle(self.canvas,      (mx, my), er, (0,0,0), -1)
                    cv2.circle(self.shape_layer, (mx, my), er, (0,0,0), -1)
                    if self.stroke_pts:
                        self._commit()
                    self.prev_x = self.prev_y = 0

                # ☝ Index only -> draw
                elif i_up and not m_up:
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = sx, sy
                    cv2.line(self.canvas,
                             (self.prev_x, self.prev_y), (sx, sy),
                             self.color, self.size)
                    self.stroke_pts.append((sx, sy))
                    self.prev_x, self.prev_y = sx, sy

                # anything else -> lift pen
                else:
                    if self.stroke_pts:
                        self._commit()
                    self.prev_x = self.prev_y = 0

            else:
                if self.stroke_pts:
                    self._commit()
                self.prev_x = self.prev_y = 0
                self.smooth_buf.clear()

            if (not bar_active and self.bar_vis
                    and time.perf_counter()-self.bar_timer < 2.0):
                self._draw_bar(img)

            # composite
            merged = cv2.add(self.canvas, self.shape_layer)
            mask   = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
            bg  = cv2.bitwise_and(img,    img,    mask=cv2.bitwise_not(mask))
            fg  = cv2.bitwise_and(merged, merged, mask=mask)
            out = cv2.add(bg, fg)

            self._hud(out, fps, hand)
            cv2.imshow("Air Draw", out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('c'), ord('C')):
                self.canvas[:]      = 0
                self.shape_layer[:] = 0
                self.history.clear()
                self.stroke_pts.clear()
            elif key in (ord('z'), ord('Z')):
                self._undo()
            elif key in (ord('s'), ord('S')):
                self.snap = not self.snap
            elif key in (ord('h'), ord('H')):
                self.show_help = not self.show_help
            elif key in (ord('+'), ord('=')):
                self.size_idx = min(self.size_idx+1, len(SIZES)-1)
                self.size     = SIZES[self.size_idx]
            elif key == ord('-'):
                self.size_idx = max(self.size_idx-1, 0)
                self.size     = SIZES[self.size_idx]

        self.alive = False
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AirDraw().run()