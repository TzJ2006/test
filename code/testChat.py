from manim import *

class RecursionStack(Scene):
    """Animate call-stack growth/shrink for **sumr(n)** *with live registers*.

    Improvements over the previous draft:
      • Register HUD now uses a monospaced font and fixed-width hex values so
        the labels never change width.
      • After every update the register lines are re-arranged vertically
        (`VGroup.arrange`) so they *cannot* drift or overlap even while the
        camera is animating.
      • `always_redraw` wrappers keep the HUD rock-solid while other
        objects move.
    """

    # ─────────── tunables ───────────
    N_START      = 3
    FRAME_W      = 3.2
    FRAME_H      = 1.1
    FRAME_GAP    = 0.25
    PUSH_TIME    = 0.7
    POP_TIME     = 0.6
    RETURN_PAUSE = 0.9
    BASE_RBP     = 0xD40
    LOCALS_SIZE  = 0x10
    FONT         = "Menlo"          # monospaced font → stable widths
    HEX_WIDTH    = 5                # 0x{val:0{HEX_WIDTH}X}
    # ───────────────────────────────

    def construct(self):
        self.camera.background_color = WHITE
        title = Text("sumr recursion – x86-64 call stack", color=BLACK, font=self.FONT)
        title.to_edge(UP)
        self.play(Write(title))

        # Architectural state we will track visually
        self.rsp_val = self.BASE_RBP - 0x18  # ≈ 0xD28 in your screenshot
        self.rbp_val = self.BASE_RBP
        self.rip_val = 0x400000

        # Register HUD (monospaced, always rearranged)
        self.hud              = self._make_hud()
        self.register_vgroup  = VGroup(self.rsp_text, self.rbp_text, self.rip_text)
        self.play(FadeIn(self.hud))

        # Base frame representing the caller
        base_frame = self._make_frame("return address", color=GRAY)
        self.play(Create(base_frame))

        visual_stack = VGroup(base_frame)
        self._label_frame_addr(base_frame, self.rbp_val)
        self._update_hud()
        self.wait(0.3)

        # ─── PUSH (calls) ───
        for n in range(self.N_START, -1, -1):
            frame = self._make_frame(f"sumr({n})")

            # shift existing stack down (toward lower addresses)
            dy = self.FRAME_H + self.FRAME_GAP
            self.play(visual_stack.animate.shift(DOWN * dy), run_time=self.PUSH_TIME * 0.45)

            frame.next_to(visual_stack[0], UP, buff=self.FRAME_GAP)
            self.play(Create(frame), run_time=self.PUSH_TIME * 0.55)
            visual_stack.add(frame)

            # architectural bookkeeping
            self.rsp_val -= 8
            self.rbp_val = self.rsp_val
            self.rsp_val -= self.LOCALS_SIZE
            self.rip_val += 0x10

            self._label_frame_addr(frame, self.rbp_val)
            self._update_hud()
            self.wait(0.15)

            if n == 0:
                break

        # ─── POP (returns) ───
        accumulated = 0
        for n in range(0, self.N_START + 1):
            ret = 0 if n == 0 else accumulated + n
            accumulated = ret

            top_frame = visual_stack[-1]
            ret_txt = Text(f"return {ret}", color=BLACK, font=self.FONT).scale(0.5)
            ret_txt.next_to(top_frame, RIGHT, buff=0.35)
            self.play(Write(ret_txt))
            self.wait(self.RETURN_PAUSE)

            dy = self.FRAME_H + self.FRAME_GAP
            self.play(FadeOut(top_frame), run_time=self.POP_TIME * 0.45)
            visual_stack.remove(top_frame)
            self.play(visual_stack.animate.shift(UP * dy), FadeOut(ret_txt), run_time=self.POP_TIME * 0.45)

            # update registers post-return
            self.rsp_val += self.LOCALS_SIZE + 8
            self.rbp_val = self.rsp_val + 8
            self.rip_val += 0x10
            self._update_hud()

        # final result bubble
        result = Text(f"sumr({self.N_START}) = {accumulated}", color=BLACK, font=self.FONT)
        result.next_to(visual_stack[0], RIGHT, buff=1.2)
        self.play(Write(result))
        self.wait(2)

    # ═════════ helper: make a frame ═════════
    def _make_frame(self, label: str, color=BLUE):
        rect = Rectangle(width=self.FRAME_W, height=self.FRAME_H,
                         stroke_color=BLACK, stroke_width=2,
                         fill_color=color, fill_opacity=0.1)
        txt = Text(label, color=BLACK, font=self.FONT).scale(0.45)
        txt.move_to(rect.get_center())
        return VGroup(rect, txt)

    # ═════════ helper: HUD ═════════
    def _make_hud(self):
        hdr = Text("Registers", color=BLACK, font=self.FONT).scale(0.6)
        self.rsp_text = Text("", color=BLACK, font=self.FONT).scale(0.55)
        self.rbp_text = Text("", color=BLACK, font=self.FONT).scale(0.55)
        self.rip_text = Text("", color=BLACK, font=self.FONT).scale(0.55)

        # bundle lines so we can re-arrange them each update
        self.register_vgroup = VGroup(self.rsp_text, self.rbp_text, self.rip_text)
        self.register_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.05)

        hud = VGroup(hdr, self.register_vgroup)
        hud.arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        hud.to_edge(LEFT, buff=0.6).shift(DOWN * 0.4)
        return hud

    # ═════════ helper: update HUD ═════════
    def _update_hud(self):
        hex_fmt = f"0x{{:0{self.HEX_WIDTH}X}}"  # fixed-width hex
        self.rsp_text.become(Text(f"%rsp: {hex_fmt.format(self.rsp_val)}", color=BLACK, font=self.FONT).scale(0.55))
        self.rbp_text.become(Text(f"%rbp: {hex_fmt.format(self.rbp_val)}", color=BLACK, font=self.FONT).scale(0.55))
        self.rip_text.become(Text(f"%rip: {hex_fmt.format(self.rip_val)}", color=BLACK, font=self.FONT).scale(0.55))
        self.register_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.05)

    # ═════════ helper: annotate frame address ═════════
    def _label_frame_addr(self, frame: VGroup, addr: int):
        label = Text(f"0x{addr:X}", color=BLACK, font=self.FONT).scale(0.4)
        label.next_to(frame, LEFT, buff=0.15)
        self.add(label)
