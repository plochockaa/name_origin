import unicodedata

# STEP 1: Normalization
def normalize_name(s):
    if not s:
        return None
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

# STEP 2: Romanization (if needed)
def romanize_if_needed(name):
    """
    Transliterate non-latin characters to latin script.
    Handles Cyrillic, Greek, and other scripts.
    """
    if not name:
        return name
    
    # Check if name contains non-latin characters
    if any(ord(c) > 127 for c in name):
        try:
            # Try to detect and transliterate Cyrillic
            if any('\u0400' <= c <= '\u04FF' for c in name):
                # Cyrillic detected - try Russian first
                try:
                    romanized = translit(name, 'ru', reversed=True)
                    return romanized
                except:
                    pass
            
            # Try unidecode as fallback for other scripts
            try:
                from unidecode import unidecode
                return unidecode(name)
            except ImportError:
                print("Warning: unidecode not installed. Install with: uv add unidecode")
                return name
                
        except Exception as e:
            print(f"Warning: Could not romanize '{name}': {e}")
            return name
    
    return name