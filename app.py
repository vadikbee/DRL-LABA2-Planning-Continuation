from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from algorithms import (
    TAXI_ENV_ID,
    mcts_parameter_study,
    taxi_policy_rows,
    td_lambda_errors,
    train_taxi,
)


st.set_page_config(
    page_title="Работа 8 | Итоговые методы",
    page_icon="8",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root { --navy:#132238; --ink:#182230; --teal:#136F63; --line:#DDE3EA; }
        .stApp { background:#FFFFFF; color:var(--ink); }
        [data-testid="stSidebar"] { background:var(--navy); border-right:0; }
        [data-testid="stSidebar"] * { color:#F7FAFC; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color:#182230; }
        [data-testid="stSidebar"] input { color:#182230; }
        h1,h2,h3 { letter-spacing:0 !important; color:var(--ink); }
        h1 { font-size:2.1rem !important; margin-bottom:.25rem !important; }
        h2 { font-size:1.35rem !important; }
        .meta { color:#5B6777; font-size:.92rem; margin-bottom:1.5rem; }
        .theory { border-left:4px solid var(--teal); padding:.25rem 1rem; margin:.8rem 0 1.2rem; }
        div[data-testid="stMetric"] { border-top:2px solid var(--line); padding-top:.7rem; }
        div[data-testid="stButton"] button { border-radius:6px; border:1px solid #0F5A51; background:#136F63; color:white; }
        div[data-testid="stButton"] button:hover { background:#0F5A51; color:white; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(
        f'<div class="meta">Баранов Вадим · № 2 по списку · {subtitle}</div>',
        unsafe_allow_html=True,
    )


def theory(text: str) -> None:
    st.markdown(f'<div class="theory">{text}</div>', unsafe_allow_html=True)


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def learning_figure(series: dict[str, np.ndarray], y_title: str) -> go.Figure:
    colors = ("#136F63", "#E76F51", "#315C8C", "#D4A72C")
    fig = go.Figure()
    for index, (label, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(values) + 1),
                y=values,
                mode="lines",
                name=label,
                line={"width": 2.2, "color": colors[index % len(colors)]},
            )
        )
    fig.update_layout(
        template="plotly_white",
        font={"color": "#182230"},
        height=450,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Эпизод / эпоха",
        yaxis_title=y_title,
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#E7EBF0")
    fig.update_yaxes(gridcolor="#E7EBF0")
    return fig


@st.cache_data(show_spinner=False)
def cached_taxi(episodes: int, simulations: int, depth: int):
    baseline = train_taxi("q_learning", episodes=episodes, seed=2)
    planned = train_taxi(
        "mcts", episodes=episodes, seed=2, simulations=simulations, depth=depth
    )
    return baseline, planned


@st.cache_data(show_spinner=False)
def cached_parameter_study(episodes: int):
    return mcts_parameter_study((5, 15, 30), (10, 25, 50), episodes=episodes)


@st.cache_data(show_spinner=False)
def cached_td(runs: int, episodes: int, alphas: tuple[float, ...], lambdas: tuple[float, ...]):
    return (
        td_lambda_errors("backward", lambdas, alphas, runs, episodes),
        td_lambda_errors("forward", lambdas, alphas, runs, episodes),
    )


@st.cache_data(show_spinner=False)
def cached_nfq(configs: tuple[tuple[str, float], ...], epochs: int, episodes_per_epoch: int):
    from nfq import train_nfq

    output = {}
    for index, (label, value) in enumerate(configs):
        kwargs = {
            "epochs": epochs,
            "episodes_per_epoch": episodes_per_epoch,
            "seed": 2 + index * 997,
        }
        if label.startswith("ε"):
            kwargs["epsilon"] = value
        elif label.startswith("batch"):
            kwargs["batch_size"] = int(value)
        elif label.startswith("γ"):
            kwargs["gamma"] = value
        else:
            kwargs["learning_rate"] = value
        output[label] = train_nfq(**kwargs)
    return output


inject_css()

with st.sidebar:
    st.markdown("## Работа 8")
    st.caption("Итоговые методы RL")
    page = st.radio(
        "Раздел",
        (
            "О работе",
            "1. ПДМК для Taxi",
            "2. TD(λ), 19 состояний",
            "3. NFQ для Taxi",
            "4. Итоговые выводы",
        ),
    )
    st.divider()
    st.caption("Исполнитель")
    st.write("Баранов Вадим")
    st.write(f"Вариант: 2 · {TAXI_ENV_ID}")

if page == "О работе":
    header("От планирования к аппроксимации", "полный отчёт по работе 8")
    theory(
        f"Вариант № 2 интерпретирован как вторая среда из списка в задании — <b>{TAXI_ENV_ID}</b>. "
        "Она дискретна, имеет 500 состояний и 6 действий, поэтому подходит и для табличного "
        "планирования, и для нейросетевой аппроксимации."
    )
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("Три эксперимента")
        st.markdown(
            """
            1. **ПДМК / MCTS:** UCT-поиск по точной модели Taxi, сравнение с Q-learning и анализ вычислительного бюджета.
            2. **TD(λ):** прямое λ-представление и обратное представление со следами приемлемости в 19-состояний случайном блуждании.
            3. **NFQ:** пакетная нейросетевая аппроксимация Q-функции и однофакторное исследование ε, batch size, γ и α.
            """
        )
    with right:
        st.subheader(f"Среда {TAXI_ENV_ID}")
        st.write(
            "Агент получает −1 за шаг, −10 за недопустимую посадку или высадку и +20 за "
            "правильную доставку. Состояние кодирует позицию такси, пассажира и пункт назначения."
        )
        st.code("state = (taxi_row, taxi_col, passenger, destination)\nactions = 6", language="python")

elif page == "1. ПДМК для Taxi":
    header("Поиск по дереву методом Монте-Карло", f"обсуждение 1 · {TAXI_ENV_ID}")
    theory(
        "ПДМК строит локальное дерево из текущего состояния. UCT балансирует среднюю ценность "
        "действия и исследовательский бонус, случайная симуляция оценивает лист, а результат "
        "распространяется обратно к корню."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        episodes = st.slider("Эпизоды обучения", 100, 800, 300, 50)
    with c2:
        simulations = st.slider("Симуляций на решение", 5, 80, 25, 5)
    with c3:
        depth = st.slider("Глубина траектории", 5, 40, 15, 5)
    with st.spinner("Обучаю базовый и планирующий агенты…"):
        baseline, planned = cached_taxi(episodes, simulations, depth)

    st.plotly_chart(
        learning_figure(
            {
                "Q-learning": moving_average(baseline.rewards),
                "Q-learning + ПДМК": moving_average(planned.rewards),
            },
            "Награда, скользящее среднее",
        ),
        theme=None,
        width="stretch",
    )
    columns = st.columns(4)
    columns[0].metric("Награда Q-learning", f"{baseline.evaluation_reward:.1f}")
    columns[1].metric("Награда ПДМК", f"{planned.evaluation_reward:.1f}")
    columns[2].metric("Успех ПДМК", f"{planned.success_rate:.0%}")
    columns[3].metric("Время ПДМК", f"{planned.duration:.2f} с")

    with st.expander("Оптимальная политика: выборка состояний", expanded=False):
        st.dataframe(
            pd.DataFrame(taxi_policy_rows(planned.q_values)),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Глубина и число симуляций")
    study_episodes = st.slider("Эпизодов на точку", 60, 240, 120, 20)
    if st.button("Запустить исследование бюджета", type="primary"):
        st.session_state["mcts_study"] = cached_parameter_study(study_episodes)
    if "mcts_study" in st.session_state:
        study = pd.DataFrame(st.session_state["mcts_study"])
        heatmap = px.density_heatmap(
            study,
            x="Симуляции",
            y="Глубина",
            z="Средняя награда",
            histfunc="avg",
            color_continuous_scale=["#E76F51", "#F4D35E", "#136F63"],
            text_auto=".1f",
        )
        heatmap.update_layout(
            template="plotly_white",
            font={"color": "#182230"},
            height=390,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
        )
        st.plotly_chart(heatmap, theme=None, width="stretch")
        st.dataframe(study.round(3), hide_index=True, width="stretch")
    st.write(
        "Увеличение глубины и числа симуляций обычно улучшает решение на раннем обучении, но "
        "время растёт примерно пропорционально их произведению. После насыщения Q-функции "
        "дополнительный поиск даёт меньший прирост."
    )

elif page == "2. TD(λ), 19 состояний":
    header("TD(λ): прямое и обратное представления", "обсуждение 2")
    theory(
        "Прямое представление смешивает n-шаговые возвраты с весами (1−λ)λⁿ⁻¹. Обратное "
        "представление обновляет посещённые состояния онлайн через следы приемлемости. В офлайн-режиме "
        "эти представления теоретически эквивалентны."
    )
    lambdas = (0.0, 0.2, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0)
    alphas = tuple(np.round(np.linspace(0.0, 0.6, 7), 3))
    c1, c2 = st.columns(2)
    with c1:
        runs = st.slider("Независимые запуски", 5, 50, 5)
    with c2:
        episodes = st.slider("Эпизоды в запуске", 3, 30, 3)
    with st.spinner("Оцениваю RMS-ошибку для всех λ и α…"):
        backward, forward = cached_td(runs, episodes, alphas, lambdas)

    selected = st.multiselect(
        "λ на графике",
        lambdas,
        default=(0.0, 0.8, 0.95, 1.0),
    )
    left, right = st.columns(2, gap="large")
    for column, title, data in (
        (left, "Обратное представление", backward),
        (right, "Прямое представление", forward),
    ):
        with column:
            fig = go.Figure()
            for lam in selected:
                fig.add_trace(go.Scatter(x=alphas, y=data[float(lam)], mode="lines+markers", name=f"λ={lam}"))
            fig.update_layout(
                template="plotly_white",
                font={"color": "#182230"},
                title=title,
                height=420,
                margin={"l": 20, "r": 20, "t": 45, "b": 20},
                xaxis_title="α",
                yaxis_title="Средняя RMS-ошибка",
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend={"orientation": "h"},
            )
            fig.update_xaxes(gridcolor="#E7EBF0")
            fig.update_yaxes(gridcolor="#E7EBF0")
            st.plotly_chart(fig, theme=None, width="stretch")

    st.caption("RMS = 5 означает расходимость; большие значения ограничены для читаемости графика.")

    rows = []
    for method_name, data in (("Обратное", backward), ("Прямое", forward)):
        for lam in lambdas:
            best_index = int(np.argmin(data[lam]))
            rows.append(
                {
                    "Представление": method_name,
                    "λ": lam,
                    "Лучшее α": alphas[best_index],
                    "RMS": data[lam][best_index],
                }
            )
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.write(
        "При λ≈0 метод близок к TD(0): смещение выше, но дисперсия мала. При λ→1 он "
        "приближается к Монте-Карло: смещение уменьшается, зато растёт чувствительность к α. "
        "Лучший компромисс в этом опыте определяется минимумом RMS в таблице."
    )

elif page == "3. NFQ для Taxi":
    header("Neural Fitted Q Iteration", f"обсуждение 3 · {TAXI_ENV_ID}")
    theory(
        "NFQ накапливает переходы, строит для них TD-цели и пакетно обучает нейросеть. "
        "Состояния Taxi кодируются one-hot-векторами длины 500, выход содержит шесть Q-оценок."
    )
    factor = st.selectbox("Исследуемый гиперпараметр", ("ε", "Размер пакета", "γ", "Скорость обучения α"))
    c1, c2 = st.columns(2)
    with c1:
        epochs = st.slider("Эпохи NFQ", 8, 40, 12, 4)
    with c2:
        per_epoch = st.slider("Эпизодов в эпохе", 8, 40, 12, 4)

    if factor == "ε":
        configs = (("ε=0.5", 0.5), ("ε=0.3", 0.3), ("ε=0.1", 0.1))
    elif factor == "Размер пакета":
        configs = (("batch=32", 32.0), ("batch=64", 64.0), ("batch=128", 128.0))
    elif factor == "γ":
        configs = (("γ=0.90", 0.90), ("γ=0.95", 0.95), ("γ=0.99", 0.99))
    else:
        configs = (("α=0.001", 0.001), ("α=0.003", 0.003), ("α=0.010", 0.010))

    with st.spinner("Обучаю три конфигурации NFQ…"):
        results = cached_nfq(configs, epochs, per_epoch)
    st.plotly_chart(
        learning_figure(
            {label: moving_average(result.rewards, 3) for label, result in results.items()},
            "Средняя награда в эпохе",
        ),
        theme=None,
        width="stretch",
    )
    table = pd.DataFrame(
        [
            {
                "Конфигурация": label,
                "Награда оценки": result.evaluation_reward,
                "Успех, %": 100 * result.success_rate,
                "Финальный loss": result.losses[-1],
                "Время, с": result.duration,
            }
            for label, result in results.items()
        ]
    ).sort_values("Награда оценки", ascending=False)
    st.dataframe(table.round(3), hide_index=True, width="stretch")
    st.subheader("Интерпретация")
    st.write(
        "Большой ε повышает разнообразие набора, но ухудшает наблюдаемую награду при сборе. "
        "Малый пакет даёт шумные градиенты, большой — более стабильные, но редкие обновления. "
        "γ определяет горизонт планирования, а α — компромисс скорости и устойчивости. Лучшей "
        "считается конфигурация по оценочной награде, а не только по минимальному loss."
    )

else:
    header("Итоговые выводы", "работа 8")
    st.markdown(
        """
        ### ПДМК
        Планирование полезно, когда модель доступна и решение нужно улучшить до накопления большого реального опыта. Цена — вычислительное время на каждом действии; глубина и число симуляций должны соответствовать горизонту задачи.

        ### TD(λ)
        λ управляет спектром между TD(0) и Монте-Карло. Прямое представление удобно для анализа возвратов, обратное — для потоковой реализации. Различия в эксперименте возникают из-за онлайн-обновлений обратного варианта.

        ### NFQ
        Нейросеть заменяет таблицу и переносит информацию между состояниями, однако появляется чувствительность к масштабу данных, скорости обучения и исследовательской политике. Сравнивать конфигурации нужно на отдельной оценочной серии.
        """
    )
    st.info(
        "Все три раздела используют зерно 2, соответствующее номеру по списку. "
        "Повторный запуск с теми же параметрами воспроизводим."
    )

st.divider()
st.caption("Работа 8 · Итоговые методы RL · Баранов Вадим · список № 2")
