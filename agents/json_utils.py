import json


def safe_json_loads(content: str):
    """
    Small local models sometimes wrap JSON in ```json fences, add stray
    preamble text, or leave trailing commentary. Try a few cheap recovery
    strategies before giving up and returning None.
    """
    content = content.strip()

    try:
        return json.loads(content)
    except Exception:
        pass

    if content.startswith("```"):
        stripped = content.strip("`")
        stripped = stripped[4:] if stripped.lower().startswith("json") else stripped
        try:
            return json.loads(stripped.strip())
        except Exception:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:
            pass

    return None
