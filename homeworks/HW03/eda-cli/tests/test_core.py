from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# тест для проверки всех 3 - х новых эвристик
# (не гуд практика, но в силу лени писать отдельные датасеты и шаблонить код я сделал 1) )
def test_quality_flags_new_heuristics():
    """Проверяет новые эвристики качества данных."""
    
    # Создаем DataFrame, в котором намеренно присутствуют проблемы:
    # 1. user_id: Дубликат (1001) -> has_suspicious_id_duplicates = True
    # 2. is_active: Категориальный (object) с 2 уникальными значениями -> has_binary_categorical_leak = True
    # 3. constant_val: Константное значение (1) -> has_constant_columns = True
    # 4. empty_col: Полностью пустая (не учитывается)

    df_with_problems = pd.DataFrame(
        {
            "user_id": [1001, 1002, 1003, 1001],
            "is_active": ["Yes", "No", "Yes", "No"], 
            "constant_val": [1, 1, 1, 1], 
            "valid_col": [4, 5, 6, 7],
            "empty_col": [None, None, None, None],
        }
    )
    
    # Считаем сводку и таблицу пропусков
    summary = summarize_dataset(df_with_problems)
    missing_df = missing_table(df_with_problems)

    flags = compute_quality_flags(summary, missing_df)
    
    # Эвристика: дубликаты в user_id
    assert flags["has_suspicious_id_duplicates"] is True, "user_id должен иметь дубликаты"
    
    # Эвристика: константная колонка
    assert flags["has_constant_columns"] is True, "Должна быть найдена константная колонка"

    # Эвристика: бинарная категориальная утечка
    assert flags["has_binary_categorical_leak"] is True, "Бинарный признак не закодирован числом"

    # Дополнительная проверка: базовый скор должен быть ниже 1.0 из-за найденных проблем
    assert flags["quality_score"] < 1.0, "quality_score должен быть снижен из-за проблем"


def test_compute_quality_flags_DF():
    """
    Проверяет ВСЕ флаги, установленные функцией compute_quality_flags, 
    на специально созданном DataFrame.
    """
    
    # Создаем DataFrame, намеренно нарушающий большинство правил:
    # - Строк: 5 (-> too_few_rows = True)
    # - Колонок: 5 (-> too_many_columns = False)
    # - Max Missing Share: 2/5 = 0.4 (-> too_many_missing = False)
    # - user_id: Дубликат (101) (-> has_suspicious_id_duplicates = True)
    # - constant_product: Константное значение (-> has_constant_columns = True)
    # - is_churned: Бинарная строка (-> has_binary_categorical_leak = True)
    df_with_all_problems = pd.DataFrame(
        {
            "user_id": [101, 102, 103, 101, 104], 
            "constant_product": ["A", "A", "A", "A", "A"], 
            "is_churned": ["Y", "N", "Y", "N", "N"], 
            "data_point_missing": [1.0, 2.0, None, 4.0, None], # 2/5 = 0.4 пропуска
            "something1": [1.0, 2.0, 3, 4.0, 5],
            "something2": [1.0, 2.0, 3, 4.0, 5],
        }
    )
    
    summary = summarize_dataset(df_with_all_problems)
    missing_df = missing_table(df_with_all_problems)
    
    flags = compute_quality_flags(summary, missing_df)

    assert flags["too_few_rows"] is True, "Строк < 100"
    assert flags["too_many_columns"] is False, "Колонок не > 100"
    assert flags["too_many_missing"] is False, "Макс. пропусков 0.4, что < 0.5"
    assert flags["max_missing_share"] == 0.4, "Макс. доля пропусков должна быть 0.4"
    
    assert flags["has_suspicious_id_duplicates"] is True, "user_id должен иметь дубликаты"
    assert flags["has_constant_columns"] is True, "Должна быть найдена константная колонка 'constant_product'"
    assert flags["has_binary_categorical_leak"] is True, "Бинарный признак не закодирован числом"

    assert 0.0 <= flags["quality_score"] <= 1.0
    assert flags["quality_score"] < 1.0, "Скор должен быть снижен из-за найденных проблем"

# Шаблонно, но всё по таскам проверено - все тесты выполняются))