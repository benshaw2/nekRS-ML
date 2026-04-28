def make_gamma_schedule(
    name="jump",
    max_val=0.5,
    delay_steps=0,
    warmup_steps=None,
    total_steps=None,
    tau=None,
):
    name = name.lower()

    if name == "constant":
        return lambda step: max_val

    if name == "linear_warmup":
        assert warmup_steps is not None
        return lambda step: max_val * min(1.0, step / warmup_steps)

    if name == "cosine":
        import math
        assert total_steps is not None
        return lambda step: max_val * 0.5 * (
            1 - math.cos(math.pi * step / total_steps)
        )

    if name == "exponential":
        import math
        assert tau is not None
        return lambda step: max_val * (1 - math.exp(-step / tau))

    if name == "jump":
        return lambda step: max_val if step >= delay_steps else 0.0

    raise ValueError(f"Unknown gamma schedule: {name}")