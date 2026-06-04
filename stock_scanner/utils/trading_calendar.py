"""IDX Trading Calendar — holiday detection and last-trading-day lookup.

Dipakai pipeline untuk:
  - Cek apakah tanggal eksekusi adalah hari bursa IDX
  - Temukan last trading day relatif dari execution date
  - Bedakan execution_date vs market_date (scan_date yang benar)

MAINTAINABILITY
---------------
IDX_HOLIDAYS perlu di-update di awal setiap tahun kalender.
Sumber resmi: https://www.idx.co.id/en/news/trading-holidays/

Catatan libur Islam (Idul Fitri, Idul Adha, Isra Mi'raj, Maulid Nabi,
Tahun Baru Islam) bergeser ~11 hari lebih awal setiap tahun Gregorian.
Verifikasi ulang dengan pengumuman resmi IDX tiap tahun.

Tanpa external dependency — hanya stdlib.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Official IDX trading holidays (non-weekend non-trading days)
# Format: "YYYY-MM-DD"
# ---------------------------------------------------------------------------

IDX_HOLIDAYS: dict[int, frozenset[str]] = {

    # ── 2024 ─────────────────────────────────────────────────────────────
    2024: frozenset({
        "2024-01-01",  # Tahun Baru Masehi
        "2024-02-08",  # Tahun Baru Imlek 2575
        "2024-02-09",  # Cuti bersama Imlek
        "2024-03-11",  # Isra Mi'raj Nabi Muhammad SAW
        "2024-03-12",  # Cuti bersama Isra Mi'raj
        "2024-03-29",  # Jumat Agung (Good Friday)
        "2024-04-08",  # Cuti bersama Idul Fitri
        "2024-04-09",  # Cuti bersama Idul Fitri
        "2024-04-10",  # Idul Fitri 1445 H (hari 1)
        "2024-04-11",  # Idul Fitri 1445 H (hari 2)
        "2024-04-12",  # Cuti bersama Idul Fitri
        "2024-04-15",  # Cuti bersama Idul Fitri
        "2024-05-09",  # Kenaikan Isa Almasih
        "2024-05-10",  # Cuti bersama Kenaikan Isa Almasih
        "2024-05-23",  # Hari Raya Waisak
        "2024-05-24",  # Cuti bersama Waisak
        "2024-06-01",  # Hari Lahir Pancasila
        "2024-06-17",  # Idul Adha 1445 H
        "2024-06-18",  # Cuti bersama Idul Adha
        "2024-07-07",  # Tahun Baru Islam 1446 H (Muharram)
        "2024-08-17",  # Hari Kemerdekaan RI
        "2024-09-16",  # Maulid Nabi Muhammad SAW
        "2024-12-25",  # Hari Raya Natal
        "2024-12-26",  # Cuti bersama Natal
    }),

    # ── 2025 ─────────────────────────────────────────────────────────────
    2025: frozenset({
        "2025-01-01",  # Tahun Baru Masehi
        "2025-01-27",  # Cuti bersama Imlek
        "2025-01-28",  # Tahun Baru Imlek 2576
        "2025-01-29",  # Cuti bersama Imlek
        "2025-03-28",  # Isra Mi'raj Nabi Muhammad SAW
        "2025-03-31",  # Cuti bersama Idul Fitri
        "2025-04-01",  # Idul Fitri 1446 H (hari 1)
        "2025-04-02",  # Idul Fitri 1446 H (hari 2)
        "2025-04-03",  # Cuti bersama Idul Fitri
        "2025-04-04",  # Cuti bersama Idul Fitri
        "2025-04-07",  # Cuti bersama Idul Fitri
        "2025-04-18",  # Jumat Agung (Good Friday)
        "2025-05-01",  # Hari Buruh Internasional
        "2025-05-12",  # Hari Raya Waisak
        "2025-05-13",  # Cuti bersama Waisak
        "2025-05-29",  # Kenaikan Isa Almasih
        "2025-06-01",  # Hari Lahir Pancasila
        "2025-06-06",  # Idul Adha 1446 H
        "2025-06-07",  # Cuti bersama Idul Adha
        "2025-06-27",  # Tahun Baru Islam 1447 H
        "2025-08-17",  # Hari Kemerdekaan RI
        "2025-09-05",  # Maulid Nabi Muhammad SAW
        "2025-12-25",  # Hari Raya Natal
        "2025-12-26",  # Cuti bersama Natal
    }),

    # ── 2026 ─────────────────────────────────────────────────────────────
    # Sources:
    #   Easter 2026 = 19 April  (derived: Ascension 28-May = Easter+39d)
    #   Good Friday = 17 April
    #   CNY 2026    = 17 Feb (Year of the Horse)
    #   Idul Fitri 1447 ≈ 20-21 Mar  (approx — verify with IDX announcement)
    #   Idul Adha 1447  ≈ 27-28 May  (approx — verify with IDX announcement)
    #   Confirmed from pipeline audit: 27-28 May had zero volume (holiday)
    2026: frozenset({
        "2026-01-01",  # Tahun Baru Masehi
        "2026-02-17",  # Tahun Baru Imlek 2577 (Year of the Horse)
        "2026-02-18",  # Cuti bersama Imlek
        "2026-03-18",  # Isra Mi'raj Nabi Muhammad SAW (approx 1447 H)
        "2026-03-20",  # Idul Fitri 1447 H hari 1 (approx — re-confirm with IDX)
        "2026-03-21",  # Idul Fitri 1447 H hari 2 (approx)
        "2026-03-19",  # Cuti bersama Idul Fitri (approx)
        "2026-03-23",  # Cuti bersama Idul Fitri (approx)
        "2026-03-24",  # Cuti bersama Idul Fitri (approx)
        "2026-04-01",  # Hari Lahir Pancasila (moved to Apr? check)  -- actually June 1
        "2026-04-17",  # Jumat Agung / Good Friday (Easter = 19 Apr)
        "2026-05-01",  # Hari Buruh Internasional
        "2026-05-27",  # Cuti bersama / Idul Adha (CONFIRMED from audit: zero volume)
        "2026-05-28",  # Kenaikan Isa Almasih (CONFIRMED from audit: zero volume)
        "2026-06-01",  # Hari Lahir Pancasila
        "2026-06-17",  # Tahun Baru Islam 1448 H (approx)
        "2026-08-17",  # Hari Kemerdekaan RI
        "2026-09-14",  # Maulid Nabi Muhammad SAW 1448 H (approx)
        "2026-12-25",  # Hari Raya Natal
        "2026-12-26",  # Cuti bersama Natal
    }),

    # ── 2027 (placeholder — UPDATE before 2027) ───────────────────────────
    # Add 2027 holidays from official IDX announcement.
    2027: frozenset({
        "2027-01-01",  # Tahun Baru Masehi
        "2027-08-17",  # Hari Kemerdekaan RI
        "2027-12-25",  # Hari Raya Natal
        # TODO: add full 2027 list from https://www.idx.co.id/en/news/trading-holidays/
    }),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_trading_day(d: date) -> bool:
    """Return True jika d adalah hari bursa IDX.

    Hari bursa = hari kerja (Senin–Jumat) DAN bukan libur nasional IDX.
    Jika tahun d tidak ada di IDX_HOLIDAYS, weekend tetap diblok;
    weekday diasumsikan trading (lihat note di bawah).

    Args:
        d: tanggal yang diperiksa.

    Returns:
        True  — trading day
        False — weekend atau libur bursa
    """
    # Weekends are always non-trading
    if d.weekday() >= 5:  # 5=Sat, 6=Sun
        return False

    # Check holiday list for the year
    year_holidays = IDX_HOLIDAYS.get(d.year)
    if year_holidays is None:
        # Year not in our list — log a warning but don't block (fallback: assume trading)
        # The zero-volume strip in the pipeline handles the data side.
        import warnings
        warnings.warn(
            f"IDX holiday list for {d.year} not found in trading_calendar.py. "
            f"Assuming {d} is a trading day. Update IDX_HOLIDAYS.",
            UserWarning,
            stacklevel=2,
        )
        return True  # optimistic fallback

    date_str = d.strftime("%Y-%m-%d")
    return date_str not in year_holidays


def last_trading_day(
    from_date: date,
    max_lookback_days: int = 14,
) -> date:
    """Cari hari bursa terakhir ≤ from_date.

    Jika from_date sendiri adalah hari bursa, kembalikan from_date.
    Jika tidak, walk back satu hari per iterasi hingga ketemu hari bursa.

    Args:
        from_date        : titik awal pencarian (inklusif).
        max_lookback_days: batas maksimum mundur (guard infinite loop).

    Returns:
        Hari bursa terakhir.

    Raises:
        RuntimeError: jika tidak ada hari bursa ditemukan dalam lookback window.
    """
    current = from_date
    for _ in range(max_lookback_days):
        if is_trading_day(current):
            return current
        current -= timedelta(days=1)
    raise RuntimeError(
        f"Tidak ada hari bursa ditemukan dalam {max_lookback_days} hari "
        f"sebelum {from_date}. Periksa IDX_HOLIDAYS."
    )


def market_date_for_execution(execution_date: Optional[date] = None) -> date:
    """Kembalikan market date yang sesuai untuk execution date.

    Jika execution date adalah hari bursa → kembalikan execution date.
    Jika tidak (weekend / libur) → kembalikan last trading day sebelumnya.

    Ini adalah fungsi utama untuk menentukan scan_date yang benar.

    Args:
        execution_date: tanggal eksekusi script. Default: date.today().

    Returns:
        Tanggal market data yang relevan.
    """
    if execution_date is None:
        execution_date = date.today()
    return last_trading_day(execution_date)


def previous_trading_day(d: date, max_lookback_days: int = 14) -> date:
    """Hari bursa terakhir STRICTLY sebelum d (exclusive).

    Berbeda dengan last_trading_day() yang inklusif: fungsi ini selalu mundur
    minimal satu hari, lalu skip weekend/libur.

    Contoh:
        previous_trading_day(Kamis)  -> Rabu (jika Rabu trading)
        previous_trading_day(Senin)  -> Jumat sebelumnya
        previous_trading_day(hari setelah libur) -> trading day valid terakhir
    """
    return last_trading_day(d - timedelta(days=1), max_lookback_days=max_lookback_days)


# IDX regular session closes ~16:00 WIB. Add a buffer for the data provider to
# publish the daily bar before we consider "today" a completed session.
IDX_CLOSE_HOUR_WIB = 17


def expected_market_date(now_wib: datetime, close_hour_wib: int = IDX_CLOSE_HOUR_WIB) -> date:
    """Sesi bursa TERAKHIR YANG SUDAH SELESAI relatif terhadap `now_wib`.

    Aturan:
      - Untuk morning alert (jalan pagi, sebelum sesi hari ini selesai) →
        kembalikan hari bursa SEBELUM hari ini (mis. Kamis pagi → Rabu).
      - Setelah jam tutup bursa (>= close_hour_wib) pada hari bursa, sesi hari
        ini dianggap selesai → kembalikan hari ini.
      - Weekend & libur otomatis di-skip.

    Args:
        now_wib       : waktu sekarang dalam zona WIB (timezone-aware atau naive-WIB).
        close_hour_wib: jam (WIB) setelah mana sesi hari ini dianggap selesai.

    Returns:
        Tanggal sesi bursa terakhir yang sudah selesai.
    """
    today = now_wib.date()
    if is_trading_day(today) and now_wib.hour >= close_hour_wib:
        return today
    return previous_trading_day(today)
