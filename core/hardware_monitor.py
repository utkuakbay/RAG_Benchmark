"""
Donanım izleme modülü - RAM/CPU takibi ve güvenlik önlemleri.
"""

import time
import gc
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil yüklü değil. Hardware monitoring devre dışı.")


@dataclass
class SystemStats:
    """Sistem istatistikleri veri sınıfı."""
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    ram_available_gb: float = 0.0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)


class HardwareMonitor:
    """
    Sistem kaynaklarını izleyen ve limit koyan sınıf.
    
    Özellikler:
    - RAM kullanımı takibi
    - CPU kullanımı takibi
    - Otomatik garbage collection
    - Threshold aşımında uyarı/durdurma
    """
    
    def __init__(
        self, 
        ram_warning_threshold: int = 85,
        ram_critical_threshold: int = 90,
        check_interval: int = 5
    ):
        """
        Args:
            ram_warning_threshold: RAM uyarı eşiği (%)
            ram_critical_threshold: RAM kritik eşiği (%) - test durdurulur
            check_interval: Kontrol aralığı (saniye)
        """
        self.ram_warning_threshold = ram_warning_threshold
        self.ram_critical_threshold = ram_critical_threshold
        self.check_interval = check_interval
        self.last_check = 0.0
        self.stats_history: list = []
        
        if not PSUTIL_AVAILABLE:
            print("⚠️ Hardware monitoring devre dışı (psutil eksik)")
    
    def get_system_stats(self) -> SystemStats:
        """
        Güncel sistem istatistiklerini al.
        
        Returns:
            SystemStats objesi
        """
        if not PSUTIL_AVAILABLE:
            return SystemStats()
        
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            stats = SystemStats(
                ram_used_gb=memory.used / (1024**3),
                ram_total_gb=memory.total / (1024**3),
                ram_percent=memory.percent,
                ram_available_gb=memory.available / (1024**3),
                cpu_percent=cpu_percent,
                timestamp=time.time()
            )
            
            # Geçmişe ekle
            self.stats_history.append(stats)
            
            # Geçmiş boyutunu sınırla (son 100 kayıt)
            if len(self.stats_history) > 100:
                self.stats_history = self.stats_history[-100:]
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Sistem istatistikleri alınamadı: {e}")
            return SystemStats()
    
    def check_resources(self, force: bool = False) -> Dict[str, Any]:
        """
        Kaynakları kontrol et, gerekirse garbage collection çalıştır.
        
        Args:
            force: Interval'i atla, hemen kontrol et
            
        Returns:
            dict: Durum bilgisi
        """
        now = time.time()
        
        # Check interval'e uymuyorsa ve force değilse atla
        if not force and (now - self.last_check) < self.check_interval:
            return {"status": "skipped", "checked": False}
        
        self.last_check = now
        stats = self.get_system_stats()
        
        result = {
            "status": "ok",
            "checked": True,
            "stats": stats,
            "warning": False,
            "critical": False
        }
        
        # Kritik eşik aşıldı mı?
        if stats.ram_percent >= self.ram_critical_threshold:
            result["status"] = "critical"
            result["critical"] = True
            print(f"\n🔴 KRİTİK: RAM %{stats.ram_percent:.1f} - Eşik: %{self.ram_critical_threshold}")
            print(f"   Kullanılan: {stats.ram_used_gb:.1f}GB / {stats.ram_total_gb:.1f}GB")
            
            # Acil garbage collection
            self._emergency_cleanup()
            
        # Uyarı eşiği aşıldı mı?
        elif stats.ram_percent >= self.ram_warning_threshold:
            result["status"] = "warning"
            result["warning"] = True
            print(f"\n⚠️ UYARI: RAM %{stats.ram_percent:.1f} - Eşik: %{self.ram_warning_threshold}")
            print(f"   Garbage collection çalıştırılıyor...")
            
            # Normal garbage collection
            freed = self._run_garbage_collection()
            result["freed_mb"] = freed
            
            print(f"   ✅ {freed:.1f}MB bellek temizlendi")
        
        return result
    
    def _run_garbage_collection(self) -> float:
        """
        Garbage collection çalıştır ve temizlenen belleği döndür.
        
        Returns:
            Temizlenen bellek (MB)
        """
        if not PSUTIL_AVAILABLE:
            gc.collect()
            return 0.0
        
        try:
            before = psutil.virtual_memory().used
            gc.collect()
            time.sleep(0.5)  # GC'nin tamamlanması için kısa bekleme
            after = psutil.virtual_memory().used
            
            freed_bytes = before - after
            freed_mb = max(0, freed_bytes / (1024**2))
            
            return freed_mb
            
        except Exception:
            gc.collect()
            return 0.0
    
    def _emergency_cleanup(self) -> None:
        """Acil bellek temizliği."""
        print("   🚨 Acil bellek temizliği başlatılıyor...")
        
        # Agresif garbage collection
        for _ in range(3):
            gc.collect()
            time.sleep(0.5)
        
        if PSUTIL_AVAILABLE:
            new_stats = self.get_system_stats()
            print(f"   Yeni RAM kullanımı: %{new_stats.ram_percent:.1f}")
    
    def wait_for_safe_ram(self, max_wait: int = 60) -> bool:
        """
        RAM güvenli seviyeye düşene kadar bekle.
        
        Args:
            max_wait: Maksimum bekleme süresi (saniye)
            
        Returns:
            True: RAM güvenli seviyede, False: Timeout
        """
        waited = 0
        
        while waited < max_wait:
            stats = self.get_system_stats()
            
            if stats.ram_percent < self.ram_warning_threshold:
                return True
            
            print(f"⏳ RAM %{stats.ram_percent:.1f} - Düşmesi bekleniyor... ({waited}s/{max_wait}s)")
            
            # Garbage collection dene
            gc.collect()
            time.sleep(5)
            waited += 5
        
        print(f"⚠️ {max_wait}s beklendi ama RAM hala yüksek!")
        return False
    
    def log_stats(self, prefix: str = "") -> None:
        """
        Mevcut istatistikleri konsola yazdır.
        
        Args:
            prefix: Log öneki
        """
        stats = self.get_system_stats()
        
        prefix_str = f"{prefix} " if prefix else ""
        
        print(f"\n📊 {prefix_str}Sistem Durumu:")
        print(f"   RAM: {stats.ram_used_gb:.1f}GB / {stats.ram_total_gb:.1f}GB (%{stats.ram_percent:.1f})")
        print(f"   CPU: %{stats.cpu_percent:.1f}")
        print(f"   Kullanılabilir RAM: {stats.ram_available_gb:.1f}GB")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        İstatistik özetini döndür.
        
        Returns:
            Özet istatistikler
        """
        if not self.stats_history:
            return {}
        
        ram_percents = [s.ram_percent for s in self.stats_history]
        cpu_percents = [s.cpu_percent for s in self.stats_history]
        
        return {
            "ram_avg": sum(ram_percents) / len(ram_percents),
            "ram_max": max(ram_percents),
            "ram_min": min(ram_percents),
            "cpu_avg": sum(cpu_percents) / len(cpu_percents),
            "cpu_max": max(cpu_percents),
            "cpu_min": min(cpu_percents),
            "samples": len(self.stats_history)
        }
    
    def is_safe_to_continue(self) -> bool:
        """
        Teste devam etmek güvenli mi?
        
        Returns:
            True: Güvenli, False: Kritik eşik aşıldı
        """
        stats = self.get_system_stats()
        return stats.ram_percent < self.ram_critical_threshold
    
    def prepare_for_next_model(self, cooldown_time: int = 3) -> None:
        """
        Sonraki model için hazırlık yap.
        
        Args:
            cooldown_time: Bekleme süresi (saniye)
        """
        print(f"\n⏳ Sonraki model için hazırlanıyor...")
        
        # Garbage collection
        gc.collect()
        
        # Bekleme
        time.sleep(cooldown_time)
        
        # Son durum
        self.log_stats("Hazırlık sonrası")
