"""
Manim animation for MACCT (https://3b1b.github.io/manim/)

manim must be installed for the appropriate OS. With and without bullets generated:

`manim -m macct_animations.py`
`MANIM_BULLET_POINTS=1 manim -m macct_animations.py` # Not yet working
"""

from manim import *


MODEL_COLOR = "#3498db"
ADAPTER_COLOR = "#2ecc71"
INACTIVE_COLOR = "#95a5a6"
VECTOR_COLOR = "#e74c3c"


def create_bullet_list(title, items, start_pos, font_size=24):
    # Create title
    title_text = Text(title, font_size=font_size).move_to(start_pos)
    
    # Create bullet points with proper spacing
    bullets = VGroup()
    for i, item in enumerate(items):
        bullet = Text(
            f"{i+1}) {item}", 
            font_size=font_size * 0.8,  # Slightly smaller font for bullets
            line_spacing=0.8
        ).next_to(
            title_text if i == 0 else bullets[-1],
            DOWN,
            aligned_edge=LEFT,
            buff=0.2
        )
        bullets.add(bullet)
    
    return VGroup(title_text, bullets)


def create_lora_adapter(height=3, width=0.5, gap=0.1, fill_color=ADAPTER_COLOR, fill_opacity=0.8):
    # Calculate dimensions for the trapezoids
    # Adjust heights to account for gap while maintaining total height
    total_height = height
    trap_height = (total_height - gap) / 2
    narrow_width = width * 0.4
    wide_width = width
    
    # Pointing down
    top_trap = Polygon(
        [-wide_width/2, trap_height/2, 0],      # Top left
        [wide_width/2, trap_height/2, 0],       # Top right
        [narrow_width/2, -trap_height/2, 0],    # Bottom right
        [-narrow_width/2, -trap_height/2, 0],   # Bottom left
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        stroke_color=WHITE
    )
    
    # Pointing up
    bottom_trap = Polygon(
        [-narrow_width/2, trap_height/2, 0],    # Top left
        [narrow_width/2, trap_height/2, 0],     # Top right
        [wide_width/2, -trap_height/2, 0],      # Bottom right
        [-wide_width/2, -trap_height/2, 0],     # Bottom left
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        stroke_color=WHITE
    )

    # Add gap at bottleneck
    top_trap.shift(UP * (trap_height/2 + gap/2))
    bottom_trap.shift(DOWN * (trap_height/2 + gap/2))
    
    lora_adapter = VGroup(top_trap, bottom_trap)
    return lora_adapter


setup_items = [
    "Load base model",
    "Freeze model weights",
    "Attach adapters A & B"
]

cycle_a_items = [
    "Generate synthetic A' from real B",
    "Switch and enable adapter B",
    "Use teacher-forcing to obtain loss\\nL(B, A')",
    "Back-propagate through adapter B"
]

cycle_b_items = [
    "Generate synthetic B' from real A",
    "Switch and enable adapter A",
    "Use teacher-forcing to obtain loss\\nL(A, B')",
    "Back-propagate through adapter A"
]


class CycleConsistencyAnimation(Scene):
    def setup_stage(self, setup_items):
        # Define the bullet points content
        setup_list = create_bullet_list("Setup", setup_items, ORIGIN, font_size=24)

        # Create base model
        self.base_model = RoundedRectangle(
            height=4,
            width=3,
            corner_radius=0.3,
            fill_color=MODEL_COLOR,
            fill_opacity=0.8,
            stroke_color=WHITE
        )
        
        # Create adapters
        self.a2b_adapter = create_lora_adapter(
            height=3,
            width=1,
            gap=0.1,
            fill_color=ADAPTER_COLOR,
            fill_opacity=0.8
        ).next_to(self.base_model, LEFT, buff=0.1)
        
        self.b2a_adapter = create_lora_adapter(
            height=3,
            width=1,
            gap=0.1,
            fill_color=INACTIVE_COLOR,
            fill_opacity=0.4
        ).next_to(self.base_model, RIGHT, buff=0.1)
        
        # Labels
        model_label = Text("Base Model").next_to(self.base_model, UP)
        a2b_label = Text("A", font_size=22).next_to(self.a2b_adapter, UP)
        b2a_label = Text("B", font_size=22).next_to(self.b2a_adapter, UP)
        
        # Snowflake for freezing effect
        snowflake = VGroup()
        for i in range(6):
            line = Line(ORIGIN, 0.3 * UP).rotate(i * TAU / 6)
            snowflake.add(line)
        snowflake.set_color(WHITE).move_to(self.base_model)

        # 1. Show base model and freeze
        self.setup_list.draw_title(self)
        self.play(FadeIn(self.base_model), FadeIn(model_label))
        self.play(
            Create(snowflake),
            Flash(self.base_model, color=WHITE, flash_radius=0.8),
            subcaption="Freeze Base Model"
        )
        self.play(
            FadeIn(self.a2b_adapter), FadeIn(a2b_label),
            FadeIn(self.b2a_adapter), FadeIn(b2a_label)
        )
        self.wait(0.5)
        self.play(FadeOut(model_label))


    def cycle_stage(self, cycle_name, cycle_items, source_label, target_label, source_adapter, target_adapter):
        # Create vector below model
        vector_source = Rectangle(
            height=0.8,
            width=2,
            fill_color=VECTOR_COLOR,
            fill_opacity=0.8
        ).next_to(self.base_model, DOWN, buff=0.5)
        vector_source_label = Text(source_label).move_to(vector_source)
        
        # Step 1: Source appears below
        self.play(FadeIn(vector_source), FadeIn(vector_source_label))
        
        # Step 2: Copy through model to target'
        vector_source_copy = vector_source.copy()
        self.play(
            vector_source_copy.animate.move_to(self.base_model.get_top() +
                vector_source_copy.get_height() * UP),
            run_time=2,
            subcaption=f"Generate synthetic outputs {target_label}' from model {target_label}→{source_label}"
        )

        vector_target_prime = vector_source_copy  # Renaming for clarity
        vector_target_prime_label = Text(f"{target_label}'").move_to(vector_target_prime)
        self.play(FadeIn(vector_target_prime_label))
        self.wait(1)

        # Adapters switch active/inactive
        self.play(
            source_adapter.animate.set_fill(color=INACTIVE_COLOR, opacity=0.4),
            target_adapter.animate.set_fill(color=ADAPTER_COLOR, opacity=0.8),
            subcaption=f"Switch Adapters to enable {target_label}→{source_label} model"
        )
        
        # Step 3: Source moves to side, target' flies around
        target_position = vector_source.get_center()
        self.play(
            vector_source.animate.shift(RIGHT * 4),
            vector_source_label.animate.shift(RIGHT * 4)
        )
        path = ArcBetweenPoints(
            vector_target_prime.get_center(),
            target_position,
            angle=TAU/2
        )
        self.play(
            MoveAlongPath(vector_target_prime, path),
            MoveAlongPath(vector_target_prime_label, path)
        )
        self.wait(0.5)

        vector_source_prime = vector_target_prime.copy()

        # Step 4: target' spawns vector through model to source'
        self.play(vector_source_prime.animate.move_to(self.base_model.get_top() + vector_source_prime.get_height() * UP), run_time=2)

        vector_source_prime_label = Text(f"{source_label}'").move_to(vector_source_prime)
        self.play(FadeIn(vector_source_prime_label))
        
        # Step 5: Move vectors up and fade target'
        self.play(
            # Move source' to the left so that the right side is centred on base model
            vector_source_prime.animate.shift(LEFT * (vector_source_prime.get_width() * 0.5 + 0.25)),
            vector_source_prime_label.animate.shift(LEFT * (vector_source_prime.get_width() * 0.5 + 0.25)),

            # Move source to the right so that the left side is centred on base model
            vector_source.animate.move_to(vector_source_prime.get_right() + 0.25 * RIGHT),
            vector_source_label.animate.move_to(vector_source_prime.get_right() + 0.25 * RIGHT),

            # Fade out target'
            FadeOut(vector_target_prime),
            FadeOut(vector_target_prime_label)
        )
        self.wait(0.5)
        
        # Step 6-7: Create loss visualization
        loss_expression = VGroup()
        loss_l = MathTex("\\mathcal{L}")
        left_bracket = Text("(")
        right_bracket = Text(")")
        
        loss_expression.add(loss_l, left_bracket)
        left_bracket.next_to(vector_source_prime, LEFT, buff=0.1)
        loss_l.next_to(left_bracket, LEFT, buff=0.1)
        right_bracket.next_to(vector_source, RIGHT, buff=0.1)
        
        self.play(Create(left_bracket), Create(right_bracket))
        self.play(Write(loss_l))
        
        # Create merging rectangle
        merge_rect = Rectangle(
            width=vector_source.get_width() * 2 + 1,
            height=0.8,
            fill_color=VECTOR_COLOR,
            fill_opacity=0.8
        ).move_to(
            (vector_source.get_center() + vector_source_prime.get_center()) / 2
        )
        
        # Merge vectors
        self.play(
            ReplacementTransform(VGroup(vector_source, vector_source_prime), merge_rect),
            FadeOut(vector_source_label),
            FadeOut(vector_source_prime_label),
            AnimationGroup(
                left_bracket.animate.next_to(merge_rect, LEFT, buff=0.1),
                loss_l.animate.next_to(merge_rect, LEFT, buff=0.3),
                right_bracket.animate.next_to(merge_rect, RIGHT, buff=0.1),
                lag_ratio=0
            )
        )

        # Shrink to point
        final_point = Dot(color=VECTOR_COLOR).move_to(merge_rect.get_center())
        
        self.play(
            ReplacementTransform(merge_rect, final_point),
            AnimationGroup(
                loss_l.animate.next_to(final_point, UP, buff=0.2),
                left_bracket.animate.next_to(final_point, LEFT, buff=0.1),
                right_bracket.animate.next_to(final_point, RIGHT, buff=0.1),
                lag_ratio=0
            )
        )
        
        # Step 8: Transform to differential
        gradient = MathTex("\\frac{\\partial \\mathcal{L}}{\\partial \\theta}").move_to(
            final_point.get_center()
        )
        self.play(
            Transform(loss_l, gradient),
            FadeOut(left_bracket),
            FadeOut(right_bracket),
            FadeOut(merge_rect)
        )
        
        # Step 9: Adapter effect and gradient flow
        gradient_arrow = Arrow(
            gradient.get_bottom(),
            target_adapter.get_top(),
            color=RED,
            buff=0.1
        )
        
        adapter_pulse = target_adapter.copy().set_color(RED)
        self.play(
            Create(gradient_arrow),
            FadeIn(adapter_pulse, rate_func=there_and_back),
            run_time=1.5
        )
        self.play(
            FadeOut(adapter_pulse),
            rate_func=there_and_back_with_pause,
            run_time=1
        )

        self.play(
            FadeOut(gradient),
            FadeOut(gradient_arrow),
            FadeOut(final_point)
        )
        self.wait(0.5)

    def construct(self):
        self.setup_stage(setup_items)
        self.wait(0.5)
        
        # Run cycle A→B
        self.cycle_stage(
            "Cycle A→B",
            cycle_a_items,
            source_label="A",
            target_label="B",
            source_adapter=self.a2b_adapter,
            target_adapter=self.b2a_adapter
        )
        
        # Run cycle B→A
        self.cycle_stage(
            "Cycle B→A",
            cycle_b_items,
            source_label="B",
            target_label="A",
            source_adapter=self.b2a_adapter,
            target_adapter=self.a2b_adapter
        )