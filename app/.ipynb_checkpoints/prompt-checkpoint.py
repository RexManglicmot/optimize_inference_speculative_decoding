# app/prompt.py

from app.config import cfg

def get_template() -> str:
    # Fallback default mirrors config
    return cfg.get("dataset", {}).get(
        "prompt_template",
        "Q: {question}\nContext: {abstract}\nA:"
    )

def render_prompt(**fields) -> str:
    """
    Render prompt text from the template and provided fields.
    Expects keys like: question=..., abstract=..., (others allowed if your template uses them).
    """
    tmpl = get_template()
    return tmpl.format(**fields)
