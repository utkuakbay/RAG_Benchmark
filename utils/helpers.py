"""
Yardımcı fonksiyonlar.
"""


def format_time(seconds: float) -> str:
    """
    Süreyi okunabilir formata çevir.
    
    Args:
        seconds: Saniye cinsinden süre
        
    Returns:
        Formatlanmış süre string'i
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_tokens(count: int) -> str:
    """
    Token sayısını okunabilir formata çevir.
    
    Args:
        count: Token sayısı
        
    Returns:
        Formatlanmış token string'i
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count/1000:.1f}K"
    else:
        return f"{count/1_000_000:.2f}M"


def format_cost(cost: float) -> str:
    """
    Maliyeti okunabilir formata çevir.
    
    Args:
        cost: USD cinsinden maliyet
        
    Returns:
        Formatlanmış maliyet string'i
    """
    if cost == 0:
        return "$0.00"
    elif cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Güvenli bölme işlemi (sıfıra bölme hatası önleme).
    
    Args:
        numerator: Pay
        denominator: Payda
        default: Payda sıfırsa döndürülecek değer
        
    Returns:
        Bölüm veya default değer
    """
    if denominator == 0:
        return default
    return numerator / denominator
