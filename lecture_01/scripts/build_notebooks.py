"""Собрать перевод и учебные дополнения, сохранив соответствие исходным ячейкам.

Запуск из корня: python lecture_01/scripts/build_notebooks.py
Повторная сборка очищает результаты выполнения сгенерированных ноутбуков.
"""

import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "references/lab_one_original.ipynb"


def clean(text):
    return dedent(text).strip() + "\n"


# Все исходные текстовые ячейки переведены; номера отсчитываются от нуля.
MARKDOWN = {
0: r"""# Лабораторная работа 1: моделирование ОДУ и СДУ""",
1: r"""Добро пожаловать в первую лабораторную! Здесь мы разберём обыкновенные и стохастические дифференциальные уравнения на интуитивных и практических примерах. Если вы заметили ошибку или хотите оставить отзыв, авторы оригинала предлагают написать на `erives@mit.edu` и `phold@mit.edu`. Приятной работы!

*Это перевод приветствия оригинала. Русский перевод, решения и дополнительные объяснения подготовлены в этом репозитории; они не являются официальным переводом MIT.*""",
3: r"""# Часть 0. Введение""",
4: r"""Сначала точно определим главные объекты: **обыкновенные дифференциальные уравнения** (ОДУ, ODE) и **стохастические дифференциальные уравнения** (СДУ, SDE). В основе обоих лежат зависящие от времени **векторные поля**. Как мы помним из лекции, это функции
$$u:\mathbb{R}^d\times[0,1]\to\mathbb{R}^d,\qquad(x,t)\mapsto u_t(x).$$
Функция $u_t(x)$ получает **наше положение в пространстве** $x$ и **момент времени** $t$, а возвращает **направление и скорость движения**. ОДУ записывается как
$$dX_t=u_t(X_t)\,dt,\qquad X_0=x_0.$$
СДУ имеет вид
$$dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t,\qquad X_0=x_0.$$
Его можно представить как ОДУ с полем $u_t$, к которому добавили шум через **броуновское движение** $(W_t)_{0\leq t\leq1}$. Детерминированную составляющую $u_t(x)$ называют **коэффициентом сноса**, а величину добавляемого шума $\sigma_t$ — **коэффициентом диффузии**.""",
6: r"""**Примечание.** ОДУ можно считать частным случаем СДУ с нулевым коэффициентом диффузии. Эта интуиция верна, но в лабораторной мы рассматриваем их отдельно ради наглядности и производительности.""",
7: r"""# Часть 1. Численные методы моделирования ОДУ и СДУ

ОДУ и СДУ можно понимать как описание движения частицы в пространстве. ОДУ говорит: «начни в $X_0=x_0$ и двигайся с мгновенной скоростью $u_t(X_t)$». СДУ добавляет к этому движению случайное возмущение, масштабируемое коэффициентом $\sigma_t$. Траектории, удовлетворяющие этим описаниям, называются **решениями** соответствующих уравнений. Численные методы приближённо находят решения путём **моделирования**, или **интегрирования**, ОДУ и СДУ.

В этой части реализуем схемы **Эйлера** для ОДУ и **Эйлера—Маруямы** для СДУ. Схема Эйлера:
$$dX_t=u_t(X_t)\,dt\quad\longrightarrow\quad X_{t+h}=X_t+h\,u_t(X_t),$$
где $h=\Delta t$ — **шаг по времени**. Схема Эйлера—Маруямы:
$$dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t\quad\longrightarrow\quad
X_{t+h}=X_t+h\,u_t(X_t)+\sqrt h\,\sigma_t z_t,\qquad z_t\sim N(0,I_d).$$
Приступим к реализации!""",
9: r"""### Задание 1.1. Реализация EulerSimulator и EulerMaruyamaSimulator""",
10: r"""**Ваша задача:** заполнить методы `step` классов `EulerSimulator` и `EulerMaruyamaSimulator`.""",
13: r"""**Примечание.** При нулевом коэффициенте диффузии схемы Эйлера и Эйлера—Маруямы дают одинаковые состояния!""",
14: r"""# Часть 2. Визуализация решений СДУ

Посмотрим, как решения СДУ выглядят на практике; к ОДУ вернёмся позднее. Реализуем и визуализируем два частных случая из лекции: **масштабированное броуновское движение** и **процесс Орнштейна—Уленбека** (OU).""",
15: r"""### Задание 2.1. Реализация броуновского движения

Напомним: броуновское движение получается при $u_t=0$ и $\sigma_t=\sigma$:
$$dX_t=\sigma\,dW_t,\qquad X_0=0.$$""",
16: r"""**Ваша задача:** интуитивно опишите траектории $X_t$ при очень большом $\sigma$. Что произойдёт при $\sigma$, близком к нулю?

**Ваш ответ:**""",
17: r"""**Ваша задача:** заполнить методы `drift_coefficient` и `diffusion_coefficient` класса `BrownianMotion`.""",
19: r"""Теперь построим графики! Для этого воспользуемся следующей вспомогательной функцией.""",
22: r"""**Ваша задача:** что происходит при изменении `sigma`?

**Ваш ответ:**""",
23: r"""### Задание 2.2. Реализация процесса Орнштейна—Уленбека

Процесс OU получается при $u_t(X_t)=-\theta X_t$ и $\sigma_t=\sigma$:
$$dX_t=-\theta X_t\,dt+\sigma\,dW_t,\qquad X_0=x_0.$$""",
24: r"""**Ваша задача:** интуитивно опишите траекторию $X_t$ при очень маленьком $\theta$. Что изменится при очень большом $\theta$?

**Ваш ответ:**""",
25: r"""**Ваша задача:** заполнить методы `drift_coefficient` и `diffusion_coefficient` класса `OUProcess`.

*В оригинале в имени второго метода была опечатка `difusion_coefficient`; здесь указано имя из кода.*""",
28: r"""**Ваша задача:** что вы замечаете о сходимости решений? Они сходятся к конкретной точке или к распределению? Ответьте двумя качественными предложениями вида: «Когда ($\theta$ или $\sigma$) увеличивается/уменьшается, мы видим…».

**Подсказка:** обратите внимание на отношение $D\triangleq\sigma^2/(2\theta)$; см. следующие ячейки.

**Ваш ответ:**""",
30: r"""**Ваша задача:** какой вывод можно сделать из предыдущего рисунка? Достаточно одного качественного предложения. Мы вернёмся к этому в разделе 3.2.

**Ваш ответ:**""",
31: r"""# Часть 3. Преобразование распределений с помощью СДУ

В предыдущей части мы наблюдали, как СДУ преобразует отдельные **точки**. Но нас интересует, как СДУ или ОДУ преобразует **распределения**. Ведь наша цель — построить уравнения, преобразующие шумовое распределение, например $N(0,I_d)$, в нужное распределение данных $p_{\mathrm{data}}$. Здесь мы визуализируем такое преобразование на примере **динамики Ланжевена**.

Сначала зададим несколько распределений для экспериментов. На практике хотелось бы иметь два свойства:

1. Уметь вычислять **плотность** $p(x)$, а значит, и градиент её логарифма $\nabla\log p(x)$. Эту величину называют **скор-функцией** (*score*); она описывает локальную геометрию распределения. С её помощью мы построим и смоделируем динамику Ланжевена — семейство СДУ, которое при подходящих условиях приводит выборку к целевому распределению $p$. В частности, динамика Ланжевена **сохраняет** распределение $p(x)$. В оригинале уточнение понятия такого движения отнесено к Lecture 2.
2. Уметь **получать выборки** из распределения $p(x)$.

Для простых учебных распределений, например гауссовских и простых смесей, обычно доступны оба свойства. Для сложных распределений, например распределения изображений, можно иметь примеры данных, но не знать плотность.

Здесь мы рассматриваем **распределения как отдельные объекты, из которых можно получать выборки**. На практике такая обёртка иногда избыточна: для простых случаев удобнее `torch.randn` и аналогичные функции.

*В оригинале целевое распределение в одном месте обозначено $\pi$, хотя вокруг используется $p$. В переводе обозначение унифицировано.*""",
36: r"""### Задание 3.1. Реализация динамики Ланжевена""",
37: r"""В этом разделе смоделируем **динамику Ланжевена без инерции**:
$$dX_t=\frac{\sigma^2}{2}\nabla\log p(X_t)\,dt+\sigma\,dW_t.$$

**Ваша задача:** заполнить методы `drift_coefficient` и `diffusion_coefficient` класса `LangevinSDE`.""",
39: r"""Теперь изобразим результаты!""",
42: r"""**Ваша задача:** изменяйте $\sigma$, число шагов и интервал времени моделирования, начальное распределение и целевую плотность. Что вы замечаете и почему?

**Ваш ответ:**""",
43: r"""**Примечание оригинала.** Для следующих двух **необязательных** ячеек нужна программа `ffmpeg`. Её можно установить, например, командой `conda install -c conda-forge ffmpeg` или через `mamba`. Команда `pip install ffmpeg` и подобные команды, скорее всего, не дадут нужного результата.

**Дополнение проекта.** Ниже анимация встроена в ноутбук через Matplotlib и HTML/JavaScript и работает без `celluloid` и `ffmpeg`. Для просмотра установите `RUN_ANIMATION = True`. Если нужен MP4, передайте `save_path="dynamics_animation.mp4"` и установите именно программу `ffmpeg`. Экспорт видео необязателен для прохождения лабораторной.""",
46: r"""### Задание 3.2. Процесс Орнштейна—Уленбека как динамика Ланжевена

Завершим работу небольшим математическим упражнением, связывающим динамику Ланжевена и процесс OU. Для достаточно регулярного распределения $p$ динамика Ланжевена имеет вид
$$dX_t=\frac{\sigma^2}{2}\nabla\log p(X_t)\,dt+\sigma\,dW_t,\qquad X_0=x_0,$$
а процесс OU при заданных $\theta,\sigma$:
$$dX_t=-\theta X_t\,dt+\sigma\,dW_t,\qquad X_0=x_0.$$""",
47: r"""**Ваша задача:** покажите, что при $p(x)=N(0,\sigma^2/(2\theta))$ скор-функция равна
$$\nabla\log p(x)=-\frac{2\theta}{\sigma^2}x.$$

**Подсказка:** плотность этого гауссовского распределения:
$$p(x)=\frac{\sqrt\theta}{\sigma\sqrt\pi}\exp\left(-\frac{x^2\theta}{\sigma^2}\right).$$

**Ваш ответ:**""",
48: r"""**Ваша задача:** сделайте вывод, что при $p(x)=N(0,\sigma^2/(2\theta))$ динамика Ланжевена
$$dX_t=\frac{\sigma^2}{2}\nabla\log p(X_t)\,dt+\sigma\,dW_t$$
эквивалентна процессу Орнштейна—Уленбека
$$dX_t=-\theta X_t\,dt+\sigma\,dW_t,\qquad X_0=0.$$

**Ваш ответ:**""",
}

ANSWERS = {
16: r"""Чем больше $\sigma$, тем шире разброс траекторий: $\operatorname{Var}(X_t)=\sigma^2t$ при $X_0=0$. При $\sigma\to0$ траектории на фиксированном интервале приближаются к нулю; при $\sigma=0$ остаются там точно. Большой шум не создаёт предпочтительного направления: среднее остаётся нулевым.""",
22: r"""При увеличении $\sigma$ вдвое стандартное отклонение $X_t$ увеличивается вдвое, а дисперсия — вчетверо. Сравните `sigma = 0.0, 0.3, 1.0, 2.0`, сохраняя `seed`, число траекторий и сетку времени. Одна траектория ничего не говорит о среднем распределения: справа показана гистограмма **всех** конечных точек.""",
24: r"""При малом положительном $\theta$ возврат к нулю слабый, и на коротком интервале процесс похож на броуновское движение. При большом положительном $\theta$ частица быстрее возвращается к нулю, а при фиксированном $\sigma$ стационарный разброс меньше. Время релаксации порядка $1/\theta$. Для большого $\theta$ нужно уменьшать шаг численной схемы.""",
28: r"""Когда $\theta>0$ увеличивается при фиксированном $\sigma>0$, распределение быстрее забывает начальное положение и его стационарная дисперсия $D=\sigma^2/(2\theta)$ уменьшается. Когда $\sigma$ увеличивается при фиксированном $\theta>0$, стационарное распределение становится шире; при $\sigma=0$ траектории детерминированно сходятся к нулю.

При $\sigma>0$ отдельная траектория продолжает колебаться: речь идёт о сходимости **распределения** к $N(0,D)$, а не о сходимости каждой частицы к точке.""",
30: r"""При одинаковом $D=\sigma^2/(2\theta)$ процессы имеют одинаковую стационарную дисперсию, хотя скорость изменения в физическом времени различается.

**Как читать этот эксперимент.** В коде время наблюдения делится на $\sigma^2$: сравнивается одинаковое масштабированное время $\tau=\sigma^2t$. Поэтому $\theta T$ и $\theta h$ одинаковы внутри строки. Верхняя строка успевает приблизиться к стационарному режиму лучше нижней: при $D=4$ ещё может быть видна память о широком начальном распределении. Совпадение формы в строке не доказывает, что стационарность уже достигнута.""",
42: r"""1. **Шум $\sigma$.** В этой параметризации растут одновременно шум и снос: снос пропорционален $\sigma^2$. Для $\sigma>0$ непрерывная динамика имеет ту же инвариантную плотность $p$, но идёт с другой скоростью. При $\sigma=0$ частицы неподвижны; сходимости к $p$ из произвольного старта нет.
2. **Число узлов.** При фиксированном конечном времени $T$ увеличение числа узлов уменьшает $h=T/(N-1)$ и ошибку дискретизации. Оно само по себе не увеличивает время перемешивания.
3. **Длительность $T$.** При сохранении малого $h$ большее $T$ помогает забыть начальное распределение. В смеси с удалёнными модами переходы между модами могут быть очень редкими: красивой картинки недостаточно, чтобы утверждать сходимость.
4. **Начальная выборка.** Если все частицы попали в один «остров» плотности, за короткое время они могут там остаться. Сравните широкую гауссовскую выборку и выборку непосредственно из целевой смеси.
5. **Целевая плотность.** Меньшее `std` делает моды уже и снос жёстче. Тогда нужен меньший шаг. Увеличение расстояния между модами затрудняет перемешивание.

Численная схема Эйлера—Маруямы с конечным шагом в общем случае **не сохраняет $p$ точно**. У неё есть ошибка дискретизации; стационарность непрерывного СДУ нельзя автоматически переносить на численную схему.""",
47: r"""Пусть $\theta>0$, $\sigma>0$. Второй аргумент $N(0,\sigma^2/(2\theta))$ здесь обозначает **дисперсию**. Логарифм плотности:
$$\log p(x)=\tfrac12\log\theta-\log\sigma-\tfrac12\log\pi-\frac{\theta x^2}{\sigma^2}.$$
Первые три слагаемых не зависят от $x$. Поэтому
$$\frac{d}{dx}\log p(x)=-\frac{2\theta}{\sigma^2}x.$$
При $\sigma=0$ распределение вырождено, и вычисление через эту плотность неприменимо.""",
48: r"""Подставляем скор-функцию в снос:
$$\frac{\sigma^2}{2}\nabla\log p(X_t)
=\frac{\sigma^2}{2}\left(-\frac{2\theta}{\sigma^2}X_t\right)=-\theta X_t.$$
Таким образом, получаем $dX_t=-\theta X_t\,dt+\sigma\,dW_t$ — процесс OU.

Эквивалентность требует одинакового начального условия; при одном и том же броуновском пути решения совпадают. В условии оригинала последняя формула использует $X_0=0$, но тот же вывод верен для любого общего $x_0$. Если $X_0=0$, распределение не является стационарным с самого начала: оно лишь стремится к $N(0,\sigma^2/(2\theta))$ при $t\to\infty$.""",
}

CODE = {
2: r'''
from abc import ABC, abstractmethod
from typing import Optional
import math
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns

# CPU достаточно для всей работы и удобно для воспроизводимости.
# Если нужна CUDA: torch.device('cuda' if torch.cuda.is_available() else 'cpu').
device = torch.device('cpu')
SEED = 2026
RUN_ANIMATION = False  # True включает необязательную анимацию.
SHOW_PROGRESS = False
MAX_PLOT_TRAJECTORIES = 60  # Гистограммы по-прежнему используют все частицы.

def set_seed(seed=SEED):
    """Повторить случайный эксперимент на том же устройстве и версиях библиотек."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
torch.set_num_threads(1)  # Для маленьких учебных тензоров это обычно быстрее.
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi': 110, 'font.family': 'DejaVu Sans'})
print(f'Устройство: {device}; PyTorch: {torch.__version__}; seed: {SEED}')
''',
5: r'''
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Снос ОДУ: состояние xt (batch_size, dim), время t (); результат той же формы, что xt."""
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Снос СДУ: состояние xt (batch_size, dim), время t (); результат (batch_size, dim)."""
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Коэффициенты независимого шума по координатам; результат (batch_size, dim)."""
        pass
''',
8: r'''
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """Один шаг: xt (batch_size, dim), скаляры t и dt; вернуть состояние при t + dt."""
        pass

    @staticmethod
    def _validate(x, ts):
        """Проверить входные данные один раз до цикла моделирования."""
        if x.ndim != 2 or not x.is_floating_point():
            raise ValueError('x должен быть вещественным тензором (batch_size, dim).')
        if ts.ndim != 1 or len(ts) == 0 or not ts.is_floating_point():
            raise ValueError('ts должен быть непустым вещественным вектором времени.')
        if x.device != ts.device or x.dtype != ts.dtype:
            raise ValueError('x и ts должны иметь одинаковые device и dtype.')
        if not torch.isfinite(ts).all() or not torch.isfinite(x).all():
            raise ValueError('Начальные состояния и времена должны быть конечными.')
        if not torch.all(ts[1:] > ts[:-1]):
            raise ValueError('Времена должны строго возрастать; обратное время здесь не поддерживается.')

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """Вернуть только финальное состояние (batch_size, dim); ts имеет форму (num_timesteps,)."""
        self._validate(x, ts)
        x = x.clone()
        for t_idx in range(len(ts) - 1):
            t = ts[t_idx]
            h = ts[t_idx + 1] - t
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """Вернуть все состояния, включая начальное: (batch_size, num_timesteps, dim)."""
        self._validate(x, ts)
        xs = [x.clone()]
        for t_idx in tqdm(range(len(ts) - 1), disable=not SHOW_PROGRESS, desc='Моделирование'):
            t = ts[t_idx]
            h = ts[t_idx + 1] - t
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
''',
11: r'''
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        # Скорость умножается на длительность шага.
        return xt + h * self.ode.drift_coefficient(xt, t)
''',
12: r'''
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        # Новый независимый стандартный нормальный шум на каждом шаге.
        z = torch.randn_like(xt)
        drift = self.sde.drift_coefficient(xt, t)
        diffusion = self.sde.diffusion_coefficient(xt, t)
        return xt + h * drift + torch.sqrt(h) * diffusion * z
''',
18: r'''
class BrownianMotion(SDE):
    def __init__(self, sigma: float):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma должна быть конечной и неотрицательной.')
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Нулевой снос, форма (batch_size, dim)."""
        return torch.zeros_like(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Постоянный коэффициент sigma, форма (batch_size, dim)."""
        return torch.full_like(xt, self.sigma)
''',
20: r'''
def plot_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor,
                         ax: Optional[Axes] = None, show_hist: bool = False,
                         decouple_hist_axis: bool = False):
    """Траектории 1D-процесса и гистограмма всех конечных точек.

    x0: (num_trajectories, 1); timesteps: (num_timesteps,).
    ax: ось для графика; decouple_hist_axis=True даёт гистограмме отдельную шкалу y.
    Возвращает все траектории для численного анализа, даже если показана только часть.
    """
    if ax is None:
        ax = plt.gca()
    trajectories = simulator.simulate_with_trajectory(x0, timesteps)
    data = trajectories.detach().cpu().numpy()
    times = timesteps.detach().cpu().numpy()
    # Ограничиваем только число линий, чтобы рисунок оставался читаемым.
    indices = np.linspace(0, len(data) - 1, min(len(data), MAX_PLOT_TRAJECTORIES), dtype=int)
    ax.plot(times, data[indices, :, 0].T, color='#177e89', alpha=0.35, linewidth=0.75)
    ax.set_xlabel('Время t')
    ax.set_ylabel(r'$X_t$')
    ax.grid(alpha=0.2)
    if show_hist:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        hist_ax = divider.append_axes('right', size='25%', pad=0.15,
                                      sharey=None if decouple_hist_axis else ax)
        hist_ax.hist(data[:, -1, 0], bins=25, orientation='horizontal', color='#dd8452', alpha=0.8)
        hist_ax.set_xlabel('Число')
        hist_ax.set_title('Финал', fontsize=9)
        hist_ax.tick_params(axis='both', labelsize=8)
        hist_ax.tick_params(axis='y', labelleft=decouple_hist_axis)
        hist_ax.grid(alpha=0.15)
    return trajectories
''',
21: r'''
set_seed()
sigma = 1.0  # Попробуйте 0.0, 0.3, 1.0, 2.0, сохраняя остальные параметры.
n_traj = 500
brownian_motion = BrownianMotion(sigma)
simulator = EulerMaruyamaSimulator(sde=brownian_motion)
x0 = torch.zeros(n_traj, 1, device=device)  # Начинаем в нуле.
ts = torch.linspace(0.0, 5.0, 500, device=device)  # 500 узлов = 499 шагов.

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_title(fr'Броуновское движение: $\sigma={sigma}$')
brownian_paths = plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True)
fig.tight_layout()
plt.show()
terminal = brownian_paths[:, -1, 0]
print(f'Среднее: {terminal.mean().item():.3f}; теория: 0')
print(f'Дисперсия: {terminal.var().item():.3f}; теория: {sigma**2 * ts[-1].item():.3f}')
''',
26: r'''
class OUProcess(SDE):
    def __init__(self, theta: float, sigma: float):
        if not math.isfinite(theta) or theta < 0:
            raise ValueError('В этой лабораторной theta должна быть конечной и неотрицательной.')
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma должна быть конечной и неотрицательной.')
        self.theta = theta
        self.sigma = sigma

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Возвращающий к нулю снос -theta * xt, форма (batch_size, dim)."""
        return -self.theta * xt

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Постоянный коэффициент диффузии, форма (batch_size, dim)."""
        return torch.full_like(xt, self.sigma)
''',
27: r'''
set_seed()
# Сначала меняем только sigma. Затем повторите с фиксированным sigma и разными theta.
thetas_and_sigmas = [(0.25, 0.0), (0.25, 0.5), (0.25, 2.0)]
simulation_time = 10.0
num_plots = len(thetas_and_sigmas)
fig, axes = plt.subplots(2, num_plots, figsize=(5 * num_plots, 7), squeeze=False)

for idx, (theta, sigma) in enumerate(thetas_and_sigmas):
    simulator = EulerMaruyamaSimulator(OUProcess(theta, sigma))
    ts = torch.linspace(0.0, simulation_time, 1000, device=device)
    if theta * float(ts[1] - ts[0]) >= 2:
        raise ValueError('Неустойчивый шаг OU: нужно theta * h < 2. Увеличьте число узлов.')
    for row, n_traj in enumerate([10, 500]):
        # Исходные точки равномерно распределены от -10 до 10, а не равны нулю.
        x0 = torch.linspace(-10.0, 10.0, n_traj, device=device).view(-1, 1)
        ax = axes[row, idx]
        ax.set_title(fr'OU: $\theta={theta}$, $\sigma={sigma}$', fontsize=11)
        plot_trajectories_1d(x0, simulator, ts, ax, show_hist=row == 1, decouple_hist_axis=True)
fig.tight_layout()
plt.show()
''',
29: r'''
set_seed()
# Сравнение процессов с одинаковой стационарной дисперсией D внутри строки.
sigmas = [1.0, 2.0, 10.0]
ds = [0.25, 1.0, 4.0]  # D = sigma**2 / (2 * theta), это не стандартное отклонение.
simulation_time = 15.0
n_traj = 500
fig, axes = plt.subplots(len(ds), len(sigmas), figsize=(15, 10), squeeze=False)
for d_idx, d in enumerate(ds):
    for s_idx, sigma in enumerate(sigmas):
        theta = sigma**2 / (2 * d)
        simulator = EulerMaruyamaSimulator(OUProcess(theta, sigma))
        x0 = torch.linspace(-20.0, 20.0, n_traj, device=device).view(-1, 1)
        time_scale = sigma**2
        ts = torch.linspace(0.0, simulation_time / time_scale, 1000, device=device)
        ax = axes[d_idx, s_idx]
        ax.set_title(fr'$\sigma={sigma:g}$, $\theta={theta:g}$, $D={d:g}$', fontsize=11)
        plot_trajectories_1d(x0, simulator, ts, ax, show_hist=True, decouple_hist_axis=True)
fig.tight_layout()
plt.show()
''',
32: r'''
class Density(ABC):
    """Распределение, для которого можно вычислить логарифм плотности."""
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, dim); вернуть логарифм плотности, форма (batch_size, 1)."""
        pass

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Градиент по x логарифма плотности, форма (batch_size, dim).

        jacrev вычисляет градиент для одной точки; vmap обрабатывает всю пачку.
        torch.func позволяет это делать и внутри симуляции под torch.no_grad().
        """
        def single_log_density(point):
            return self.log_density(point.unsqueeze(0)).squeeze()
        return vmap(jacrev(single_log_density))(x)

class Sampleable(ABC):
    """Распределение, из которого можно получать выборки."""
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """Получить num_samples точек; форма результата (num_samples, dim)."""
        pass
''',
33: r'''
def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    """Двумерная гистограмма выборки; kwargs передаются в hist2d."""
    ax = plt.gca() if ax is None else ax
    samples = sampleable.sample(num_samples).detach().cpu().numpy()
    return ax.hist2d(samples[:, 0], samples[:, 1], **kwargs)

def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    """Точечный график выборки; kwargs передаются в scatter."""
    ax = plt.gca() if ax is None else ax
    samples = sampleable.sample(num_samples).detach().cpu().numpy()
    return ax.scatter(samples[:, 0], samples[:, 1], **kwargs)

def density_grid(density: Density, bins: int, scale: float):
    """Сетка логарифма плотности для двумерных рисунков."""
    x = torch.linspace(-scale, scale, bins, device=device)
    y = torch.linspace(-scale, scale, bins, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        return density.log_density(xy).reshape(bins, bins).T.detach().cpu().numpy()

def imshow_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, **kwargs):
    """Тепловая карта log p(x), а не самой плотности p(x)."""
    ax = plt.gca() if ax is None else ax
    return ax.imshow(density_grid(density, bins, scale), extent=[-scale, scale, -scale, scale],
                     origin='lower', **kwargs)

def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, **kwargs):
    """Линии уровня log p(x)."""
    ax = plt.gca() if ax is None else ax
    return ax.contour(density_grid(density, bins, scale), extent=[-scale, scale, -scale, scale],
                      origin='lower', **kwargs)
''',
34: r'''
class Gaussian(torch.nn.Module, Sampleable, Density):
    """Гауссовское распределение: обёртка над MultivariateNormal; примеры используют 2D."""
    def __init__(self, mean, cov):
        """mean: (dim,); cov: положительно определённая ковариация (dim, dim)."""
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """Смесь гауссовских распределений с выборками и вычислимой плотностью."""
    def __init__(self, means: torch.Tensor, covs: torch.Tensor, weights: torch.Tensor):
        """means: (nmodes, dim); covs: (nmodes, dim, dim); weights: (nmodes,)."""
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer('means', means)
        self.register_buffer('covs', covs)
        self.register_buffer('weights', weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    @classmethod
    def random_2D(cls, nmodes: int, std: float, scale: float = 10.0, seed: int = 0):
        """Случайные центры; отдельный генератор не сбрасывает общий поток случайных чисел."""
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        means = (torch.rand(nmodes, 2, generator=generator) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, std: float, scale: float = 10.0):
        """Центры на окружности радиуса scale; стандартное отклонение каждой моды std."""
        angles = torch.linspace(0, 2 * math.pi, nmodes + 1)[:-1]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)
''',
35: r'''
densities = {
    'Гауссовское распределение': Gaussian(torch.zeros(2), 10 * torch.eye(2)).to(device),
    'Случайная смесь': GaussianMixture.random_2D(5, std=1.0, scale=20.0, seed=3).to(device),
    'Симметричная смесь': GaussianMixture.symmetric_2D(5, std=1.0, scale=8.0).to(device),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
for ax, (name, density) in zip(axes, densities.items()):
    ax.set_title(name)
    im = imshow_density(density, 100, 15, ax, vmin=-15, vmax=0, cmap='Blues')
    contour_density(density, 100, 15, ax, colors='grey', linestyles='solid', alpha=0.25, levels=20)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
fig.colorbar(im, ax=axes, label='Логарифм плотности log p(x)', shrink=0.8)
plt.show()
''',
38: r'''
class LangevinSDE(SDE):
    def __init__(self, sigma: float, density: Density):
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError('sigma должна быть конечной и неотрицательной.')
        self.sigma = sigma
        self.density = density

    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Снос (sigma**2 / 2) * score; форма (batch_size, dim)."""
        return 0.5 * self.sigma**2 * self.density.score(xt)

    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Независимый шум с одинаковым коэффициентом sigma по всем координатам."""
        return torch.full_like(xt, self.sigma)
''',
40: r'''
def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """Индексы через n узлов; первый и последний обязательно входят без повторов."""
    if num_timesteps < 1 or n < 1:
        raise ValueError('Число узлов и интервал записи должны быть положительными.')
    indices = list(range(0, num_timesteps, n))
    if indices[-1] != num_timesteps - 1:
        indices.append(num_timesteps - 1)
    return torch.tensor(indices, dtype=torch.long)

def graph_dynamics(num_samples: int, source_distribution: Sampleable, simulator: Simulator,
                   density: Density, timesteps: torch.Tensor, plot_every: int,
                   bins: int, scale: float):
    """Эволюция выборки: сверху точки, снизу оценка плотности по выборке (KDE).

    source_distribution задаёт старт; density — целевую плотность фона.
    timesteps задаёт сетку; plot_every — промежуток между снимками; bins — разрешение
    фона; scale — половина стороны окна. Возвращает полную траекторию.
    """
    x0 = source_distribution.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, timesteps)
    indices = every_nth_index(len(timesteps), plot_every).to(timesteps.device)
    times = timesteps[indices].cpu().numpy()
    snapshots = xts[:, indices].cpu().numpy()
    background = density_grid(density, bins, scale)
    fig, axes = plt.subplots(2, len(times), figsize=(4 * len(times), 7), squeeze=False)
    for col, time in enumerate(times):
        points = snapshots[:, col]
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(background, extent=[-scale, scale, -scale, scale], origin='lower',
                      vmin=-15, vmax=0, cmap='Blues', alpha=0.35)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
        axes[0, col].scatter(points[:, 0], points[:, 1], s=5, c='#222222', alpha=0.3)
        axes[0, col].set_title(f'Точки, t={time:.2f}')
        sns.kdeplot(x=points[:, 0], y=points[:, 1], ax=axes[1, col], color='#a64b24', levels=6)
        axes[1, col].set_title(f'Оценка плотности, t={time:.2f}')
    fig.tight_layout()
    plt.show()
    return xts
''',
41: r'''
set_seed()
target = GaussianMixture.random_2D(5, std=0.75, scale=15.0, seed=3).to(device)
sde = LangevinSDE(sigma=0.6, density=target)
simulator = EulerMaruyamaSimulator(sde)
langevin_paths = graph_dynamics(
    num_samples=1000,
    source_distribution=Gaussian(torch.zeros(2), 20 * torch.eye(2)).to(device),
    simulator=simulator,
    density=target,
    timesteps=torch.linspace(0, 5.0, 1000, device=device),
    plot_every=334,
    bins=200,
    scale=15,
)
print('Форма траекторий:', tuple(langevin_paths.shape))
''',
44: r'''
from matplotlib import animation
from IPython.display import HTML, display

def animate_dynamics(num_samples: int, source_distribution: Sampleable, simulator: Simulator,
                     density: Density, timesteps: torch.Tensor, animate_every: int,
                     bins: int, scale: float, save_path: Optional[str] = None):
    """Анимация точек и KDE; HTML без внешнего видеокодека, необязательный экспорт MP4.

    animate_every задаёт число узлов между кадрами; остальные параметры как у graph_dynamics.
    """
    if save_path and not animation.writers.is_available('ffmpeg'):
        raise RuntimeError('Для MP4 установите программу ffmpeg или используйте save_path=None.')
    x0 = source_distribution.sample(num_samples)
    paths = simulator.simulate_with_trajectory(x0, timesteps)
    indices = every_nth_index(len(timesteps), animate_every).to(timesteps.device)
    snapshots = paths[:, indices].cpu().numpy()
    times = timesteps[indices].cpu().numpy()
    background = density_grid(density, bins, scale)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')

    def update(frame):
        points = snapshots[:, frame]
        for ax in axes:
            ax.clear()
            ax.imshow(background, extent=[-scale, scale, -scale, scale], origin='lower',
                      vmin=-15, vmax=0, cmap='Blues', alpha=0.35)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
        axes[0].scatter(points[:, 0], points[:, 1], s=6, c='#222222', alpha=0.4)
        sns.kdeplot(x=points[:, 0], y=points[:, 1], ax=axes[1], color='#a64b24', levels=5)
        axes[0].set_title(f'Точки, t={times[frame]:.2f}')
        axes[1].set_title(f'Оценка плотности, t={times[frame]:.2f}')

    movie = animation.FuncAnimation(fig, update, frames=len(times), interval=180, repeat=True)
    try:
        if save_path:
            movie.save(save_path, writer='ffmpeg', fps=6)
        return HTML(movie.to_jshtml(fps=6))
    finally:
        plt.close(fig)
''',
45: r'''
# Необязательный этап: установите RUN_ANIMATION = True и выполните эту ячейку.
if RUN_ANIMATION:
    set_seed()
    animation_target = GaussianMixture.random_2D(5, std=0.75, scale=15.0, seed=3).to(device)
    display(animate_dynamics(
        num_samples=1000,
        source_distribution=Gaussian(torch.zeros(2), 20 * torch.eye(2)).to(device),
        simulator=EulerMaruyamaSimulator(LangevinSDE(0.6, animation_target)),
        density=animation_target,
        timesteps=torch.linspace(0, 5.0, 1000, device=device),
        animate_every=100,
        bins=200,
        scale=15,
    ))
else:
    print('Анимация выключена. Установите RUN_ANIMATION = True и повторите эту ячейку.')
''',
}

# Подсказки к каждому этапу добавляются сверх полного перевода оригинала.
BEFORE = {
2: r"""## Как работать с этим ноутбуком

1. Используйте Python 3.11/3.12 и установите зависимости по `README.md`. В Google Colab загрузите этот файл через **File → Upload notebook**; при нехватке пакетов выполните `%pip install torch numpy matplotlib seaborn scipy tqdm` в отдельной ячейке.
2. Выполняйте ячейки сверху вниз с **Shift+Enter**. Для проверки с чистого состояния выберите **Kernel → Restart Kernel and Run All Cells**. Состояние хранится в ядре: определение класса само по себе ничего не рисует.
3. Прочитайте условие, запустите реализацию, рассмотрите график, затем измените **один** параметр и повторите эксперимент. При переопределении класса заново выполните ячейки, создающие его экземпляры.
4. Ответы и дополнительные эксперименты помечены **«Разбор»** и **«Дополнение проекта»**. В версии для самостоятельной работы решения скрыты заменой на заготовки.
5. Для воспроизводимости повторяйте `set_seed()` перед сравниваемыми запусками. Для независимого повторения меняйте `SEED` и вызывайте `set_seed(SEED)` явно.

### Маршрут

| Этап | Что делаем | Что должно получиться |
|---|---|---|
| 0 | Разбираем `ODE`, `SDE`, формы тензоров | Понимаем снос и диффузию |
| 1.1 | Пишем `step` двух схем | ОДУ без шума; СДУ с шумом масштаба $\sqrt h$ |
| 2.1 | Реализуем броуновское движение | Разброс растёт как $\sigma\sqrt t$ |
| 2.2 | Добавляем возврат к нулю | OU и сравнение $D=\sigma^2/(2\theta)$ |
| 3.1 | Вычисляем score и запускаем Ланжевена | Выборка перемещается к областям высокой плотности |
| Анимация | Включаем `RUN_ANIMATION` | Кадры преобразования выборки |
| 3.2 | Подставляем score гауссианы | Получаем снос $-\theta x$ |

**Источники:** [курс MIT 6.S184 (2026)](https://diffusion.csail.mit.edu/2026/), [Lecture 1](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf), предоставленный пользователем `lab_one.ipynb` (его точная копия лежит в `references`). Авторы курса: Peter Holderrieth, Ron Shprints, Ezra Erives. Атрибуция и изменения описаны в `references/README.md`.

Здесь изучаются численные методы и выборки из известных учебных плотностей. Обучение нейросети и получение неизвестного $p_{\mathrm{data}}$ потребуют следующих лекций.""",
5: r"""### Дополнение проекта: как читать интерфейсы

`xt` — пачка точек формы `(batch_size, dim)`. Например, `(500, 1)` — 500 частиц на прямой, `(1000, 2)` — 1000 частиц на плоскости. `t` и `h` — скаляры (тензоры формы `()`).

`drift_coefficient` возвращает скорость для каждой точки. `diffusion_coefficient` в этой работе возвращает коэффициенты независимого шума по координатам той же формы, что `xt`; это диагональный шум, а не произвольная матрица диффузии. `ABC` задаёт интерфейс, `@abstractmethod` требует реализацию в потомке.

Интервал $[0,1]$ в определении можно заменить на $[0,T]$; поэтому далее встречаются $T=5,10,15$. Слово «скорость» для броуновской части лишь интуиция: обычной производной у её траекторий почти наверное нет.""",
11: r"""### Дополнение проекта: от формулы к строкам кода

1. Получите снос методом `drift_coefficient(xt, t)`.
2. Умножьте его на `h` и прибавьте к текущему состоянию.
3. Для СДУ отдельно получите коэффициент диффузии и `torch.randn_like(xt)`.
4. Умножьте шум на `torch.sqrt(h)`: приращение $W_{t+h}-W_t$ имеет дисперсию $h$, а не $h^2$.

`simulate` хранит только последний результат. `simulate_with_trajectory` хранит и начальную точку, поэтому для `N` узлов делает `N-1` шагов и возвращает форму `(batch_size, N, dim)`. Чем больше частиц и узлов, тем больше памяти нужно.""",
21: r"""### Дополнение проекта: эксперимент с броуновским движением

Сначала запустите `sigma=1.0`. По горизонтали идёт время, по вертикали — положение частицы. Справа — число частиц в интервалах конечного положения. На основном рисунке показано до 60 траекторий, а гистограмма и численные характеристики используют все 500.

Затем повторите с `sigma=0.0`, `0.3` и `2.0`. Сравнивайте **числа**, поскольку Matplotlib автоматически меняет пределы осей. Для точного визуального сравнения зафиксируйте одинаковые `ax.set_ylim(...)` до показа рисунка. Конечная выборочная дисперсия лишь приблизительно равна теоретической: 500 случайных частиц дают статистическую погрешность.""",
27: r"""### Дополнение проекта: как менять параметры OU

В первой сетке верхний ряд показывает 10 траекторий, нижний — траектории и гистограмму для 500 частиц. Сначала меняется только шум. Затем замените список на `[(0.1, 1.0), (0.5, 1.0), (2.0, 1.0)]`, чтобы менять только силу возврата.

Для $\theta>0$ и фиксированного $x_0$:
$$\mathbb E[X_t]=x_0e^{-\theta t},\qquad
\operatorname{Var}(X_t)=\frac{\sigma^2}{2\theta}(1-e^{-2\theta t}).$$
При случайном $X_0$ к дисперсии добавляется $e^{-2\theta t}\operatorname{Var}(X_0)$, если старт независим от будущего шума.

Для явной схемы Эйлера нужно $0<\theta h<2$, чтобы возвращающая часть была устойчивой. Для точности выбирайте $\theta h\ll1$. При $\theta=0$ OU превращается в броуновское движение, и стационарная формула $D$ неприменима. В гистограммах ниже может быть своя шкала положения — читайте подписи делений.""",
32: r"""### Дополнение проекта: плотность, выборка и score

`sample(1000)` возвращает **точки**, `log_density(x)` — **значения логарифма плотности**, а `score(x)` — **векторы направления её роста**. Это три разных операции.

Здесь score считается автоматическим дифференцированием известной плотности, без обучения нейросети. Для реальных изображений такой точной формулы обычно нет.

Классы наследуют `torch.nn.Module`, чтобы `.to(device)` переносил параметры распределения. `register_buffer` хранит средние, ковариации и веса как тензоры, которые не являются обучаемыми параметрами.""",
38: r"""### Дополнение проекта: шаг динамики Ланжевена

Плотность задаёт направление через score. Метод `drift_coefficient` должен умножить его на $\sigma^2/2$, а `diffusion_coefficient` возвращает $\sigma$ по всем координатам.

**Проверка смысла:** в гауссовском распределении score направлен к среднему; шум не даёт всем точкам просто «свалиться» в максимум. Для сохранения нужной плотности соотношение коэффициентов сноса и шума принципиально.""",
41: r"""### Дополнение проекта: как читать преобразование распределения

Каждый столбец — момент времени. Верхний ряд — частицы, нижний — сглаженная оценка их плотности (KDE). Синий фон — **логарифм целевой плотности**, одинаковый во всех столбцах. KDE зависит от числа точек и степени сглаживания; это не точная плотность процесса.

Порядок экспериментов: сначала исходные настройки, затем `sigma=1.2`; после этого увеличьте число узлов при прежнем $T=5$. Для отдельной проверки влияния времени увеличьте $T$ и число узлов пропорционально, сохранив шаг. Затем меняйте `std` и `scale` смеси.

Цель первого запуска — увидеть движение к модам. При $T=5$ и $\sigma=0.6$ совпадение конечного распределения с целевой смесью не гарантируется. Для надёжного контроля ниже есть количественный эксперимент с одной гауссианой, где известен точный ответ.""",
46: r"""### Дополнение проекта: как выполнить математический этап

Сначала возьмите логарифм плотности, затем производную по $x$, после этого умножьте на $\sigma^2/2$. Параметры $\theta$ и $\sigma$ при дифференцировании считаются константами. Условия для невырожденной стационарной гауссианы: $\theta>0$, $\sigma>0$.""",
}

ODE_DEMO = r'''
class LinearODE(ODE):
    """Пример ОДУ dX/dt = -theta * X с известным точным решением."""
    def __init__(self, theta=1.0):
        self.theta = theta

    def drift_coefficient(self, xt, t):
        return -self.theta * xt

x0_ode = torch.tensor([[2.0]], device=device)
fig, ax = plt.subplots(figsize=(8, 4))
fine_ts = torch.linspace(0, 3, 301, device=device)
ax.plot(fine_ts.cpu(), (2 * torch.exp(-fine_ts)).cpu(), 'k--', label='Точное решение')
ode_errors = []
for nodes in [7, 31, 151]:
    ode_ts = torch.linspace(0, 3, nodes, device=device)
    path = EulerSimulator(LinearODE()).simulate_with_trajectory(x0_ode, ode_ts)
    error = abs(path[0, -1, 0].item() - 2 * math.exp(-3))
    ode_errors.append(error)
    ax.plot(ode_ts.cpu(), path[0, :, 0].cpu(), label=f'{nodes-1} шагов; ошибка {error:.4f}')
ax.set(title='Метод Эйлера и точное решение ОДУ', xlabel='Время t', ylabel=r'$X_t$')
ax.legend()
fig.tight_layout()
plt.show()
assert ode_errors[2] < ode_errors[1] < ode_errors[0]
print('Проверка пройдена: уменьшение шага уменьшает ошибку в этом примере.')
'''

SCORE_DEMO = r'''
gaussian_check = Gaussian(torch.zeros(2), 2 * torch.eye(2)).to(device)
probe_points = torch.tensor([[2.0, -4.0], [0.0, 0.0]], device=device)
computed_score = gaussian_check.score(probe_points)
expected_score = -probe_points / 2  # Для N(0, 2I): score = -x / 2.
print('Точки:\n', probe_points)
print('Скор-функция:\n', computed_score)
assert torch.allclose(computed_score, expected_score, atol=1e-6)
print('Проверка пройдена: автоматический градиент совпадает с формулой.')
'''

GAUSSIAN_DEMO = r'''
set_seed()
gaussian_target = Gaussian(torch.zeros(2), torch.eye(2)).to(device)
gaussian_sde = LangevinSDE(sigma=math.sqrt(2), density=gaussian_target)
# В этом случае снос равен -x: OU с theta=1, sigma=sqrt(2).
moment_ts = torch.linspace(0, 5, 1001, device=device)
moment_x0 = torch.full((5000, 2), 3.0, device=device)
moment_final = EulerMaruyamaSimulator(gaussian_sde).simulate(moment_x0, moment_ts)
exact_mean = 3 * math.exp(-5)
exact_var = 1 - math.exp(-10)
print('Выборочное среднее:', moment_final.mean(0).cpu().numpy())
print('Точное среднее непрерывного СДУ:', round(exact_mean, 5))
print('Выборочная дисперсия:', moment_final.var(0).cpu().numpy())
print('Точная дисперсия непрерывного СДУ:', round(exact_var, 5))
print('Допуски учитывают случайную выборку и конечный шаг h=0.005.')
assert torch.all((moment_final.mean(0) - exact_mean).abs() < 0.07)
assert torch.all((moment_final.var(0) - exact_var).abs() < 0.09)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(moment_final[:, 0].cpu().numpy(), bins=45, density=True, alpha=0.55,
        color='#177e89', label='Конечная выборка, координата 1')
grid = np.linspace(-4, 4, 300)
pdf = np.exp(-0.5 * (grid - exact_mean)**2 / exact_var) / np.sqrt(2 * np.pi * exact_var)
ax.plot(grid, pdf, color='#a64b24', label='Точная плотность при t=5')
ax.set(xlabel='Положение x', ylabel='Плотность', title='Ланжевен для одной гауссианы: количественный контроль')
ax.legend()
fig.tight_layout()
plt.show()
'''


def make_cell(kind, text, cell_id, **metadata):
    result = {"cell_type": kind, "id": cell_id, "metadata": metadata,
              "source": clean(text).splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def exercise_code(text):
    """Заменить только учебные методы, сохранив весь вспомогательный код."""
    import ast
    tree = ast.parse(clean(text))
    lines = clean(text).splitlines(keepends=True)
    targets = {"step", "drift_coefficient", "diffusion_coefficient"}
    edits = []
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and method.name in targets:
                    edits.append((method.body[0].lineno - 1, method.end_lineno))
    for start, stop in sorted(edits, reverse=True):
        lines[start:stop] = ['        raise NotImplementedError("Реализуйте метод по условию задания.")\n']
    return ''.join(lines)


def build(exercises=False):
    original = json.loads(ORIGINAL.read_text())
    assert len(original['cells']) == 49, 'Изменился исходник: пересмотрите карту перевода.'
    assert set(MARKDOWN) == {i for i,c in enumerate(original['cells']) if c['cell_type']=='markdown'}
    assert set(CODE) == {i for i,c in enumerate(original['cells']) if c['cell_type']=='code'}
    cells = []
    definition_indices = {2, 5, 8, 11, 12, 18, 20, 26, 32, 33, 34, 38, 40, 44}
    for index, original_cell in enumerate(original['cells']):
        if index in BEFORE:
            cells.append(make_cell('markdown', BEFORE[index], f'guide-{index:02}', role='guide'))
        kind = original_cell['cell_type']
        source = (MARKDOWN if kind == 'markdown' else CODE)[index]
        if index == 0:
            source += '\n\n' + ('**Версия для самостоятельной работы.** Заполните методы и ответы. До этого запуск всех ячеек остановится на незавершённом задании.' if exercises else '**Версия с решениями и пошаговым разбором.** Все основные ячейки готовы к выполнению.')
        if exercises and index in {11, 12, 18, 26, 38}:
            source = exercise_code(source)
        tags = ['definitions'] if index in definition_indices else []
        if index == 45:
            tags.append('optional-animation')
        cells.append(make_cell(kind, source, f'original-{index:02}', original_cell_index=index, tags=tags))
        if not exercises and index in ANSWERS:
            cells.append(make_cell('markdown', '**Разбор (дополнение проекта).**\n\n' + ANSWERS[index], f'answer-{index:02}', role='solution'))
        if index == 13:
            cells.append(make_cell('markdown', r'''### Дополнение проекта: проверяем ОДУ на точном решении

Для $dX_t=-X_t\,dt$, $X_0=2$ точное решение — $X_t=2e^{-t}$. Запустите следующий пример после задания 1.1. Сравните 6, 30 и 150 шагов на одном интервале. Меняется ошибка численного метода, а не само уравнение.''', 'ode-guide', role='guide'))
            cells.append(make_cell('code', ODE_DEMO, 'ode-demo', tags=['experiment']))
        if index == 35:
            cells.append(make_cell('markdown', '### Дополнение проекта: проверяем скор-функцию гауссианы\n\nДля $N(\\mu,\\Sigma)$ она равна $-\\Sigma^{-1}(x-\\mu)$. Сравним вычисление `jacrev` с точной формулой.', 'score-guide', role='guide'))
            cells.append(make_cell('code', SCORE_DEMO, 'score-demo', tags=['experiment']))
        if index == 48:
            cells.append(make_cell('markdown', '### Дополнение проекта: проверяем динамику по среднему и дисперсии\n\nЭтот эксперимент использует одну гауссиану и фиксированную начальную точку. Поэтому можно сравнить численное решение с точным распределением OU при конечном времени, не делая предположений о перемешивании смеси.', 'moments-guide', role='guide'))
            cells.append(make_cell('code', GAUSSIAN_DEMO, 'moments-demo', tags=['experiment']))
    cells.append(make_cell('markdown', r'''## Итог и проверка понимания

- Объясните, почему шум масштабируется через $\sqrt h$.
- Покажите убывание ошибки Эйлера на ОДУ с точным решением.
- Отличите рост дисперсии броуновского движения от стационарного разброса OU.
- Объясните разницу между `sample`, `log_density` и `score`.
- Покажите, как меняются результаты Ланжевена при изменении шага и времени по отдельности.
- Выведите OU из динамики Ланжевена и объясните условия $\theta>0$, $\sigma>0$.

**Мини-отчёт:** для каждого этапа сохраните параметры, рисунок, теоретическое ожидание, наблюдение и возможную причину расхождения. Не ограничивайтесь фразой «получилось похоже».

**Связь с остальным репозиторием.** Ноутбук `Diffusion_DDPM_Research.ipynb` изучает добавление и удаление шума на CIFAR-10 с нейросетями. Здесь подготовлены основы непрерывного времени и численного семплирования. Следующий шаг курса — научиться задавать подходящие поля с помощью обучения.

**Готовность:** перезапустите ядро и выполните все ячейки сверху вниз. В версии с решениями должны появиться все статические графики и пройти численные проверки; анимация включается отдельно. Дополнительные проверки можно запустить из корня репозитория: `python -m pytest lecture_01/tests -q`.''', 'completion', role='guide'))
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name':'Python 3 (ipykernel)', 'language':'python', 'name':'python3'},
            'language_info': {'name':'python','version':'3.12','file_extension':'.py','mimetype':'text/x-python','pygments_lexer':'ipython3','nbconvert_exporter':'python'},
            'lab_translation': {'language':'ru', 'original_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                                'original_cells':49, 'variant':'exercises' if exercises else 'walkthrough',
                                'course_url':'https://diffusion.csail.mit.edu/2026/'},
        },
        'nbformat':4, 'nbformat_minor':5,
    }


if __name__ == '__main__':
    for exercises, name in [(False, 'lab_one_ru.ipynb'), (True, 'lab_one_ru_exercises.ipynb')]:
        notebook = build(exercises)
        (ROOT / name).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
        print(f'{name}: {len(notebook["cells"])} ячеек, 49 исходных ячеек учтены.')
