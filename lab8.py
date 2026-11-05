"""ЛР №8: CBOW (Word Embeddings), NumPy реалізація з мінібатчним SGD.

Власна реалізація прямого/зворотного поширення: без готової Word2Vec моделі.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
EXAMPLES = ['king', 'queen', 'lord', 'man', 'woman', 'dog', 'wolf', 'rich', 'happy', 'sad', 'love', 'war', 'prince', 'england', 'france']


def tokenize(text: str) -> list[str]:
    """Слова нижнього регістру, без пунктуації, зі збереженням апострофів."""
    return TOKEN_PATTERN.findall(text.lower())


def load_sentences(path: str | Path, *, min_words: int = 5) -> list[list[str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [tok for line in f if len(tok := tokenize(line)) >= min_words]


def make_vocab(sentences: list[list[str]], min_freq: int = 3) -> tuple[dict[str, int], Counter]:
    counts = Counter(word for sent in sentences for word in sent)
    vocabulary = ['<unk>'] + sorted(w for w, count in counts.items() if count >= min_freq)
    return {word: i for i, word in enumerate(vocabulary)}, counts


def create_examples(sentences: list[list[str]], word2idx: dict[str, int], C: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Матриця індексів контексту (n, 2*C), індекси центральних слів (n,)."""
    if C < 1:
        raise ValueError('C must be >= 1')
    x, y = [], []
    unk = word2idx['<unk>']
    for sent in sentences:
        indices = [word2idx.get(word, unk) for word in sent]
        for i in range(C, len(indices) - C):
            x.append(indices[i-C:i] + indices[i+1:i+C+1]); y.append(indices[i])
    if not x:
        raise ValueError('Недостатньо довгих речень для контекстного вікна')
    return np.asarray(x, dtype=np.int32), np.asarray(y, dtype=np.int32)


def initialize_model(n_words: int, dim: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 0.12
    return {
        'W1': rng.normal(0, scale, size=(dim, n_words)).astype(np.float32),
        'W2': rng.normal(0, scale, size=(n_words, dim)).astype(np.float32),
        'b1': np.full(dim, .04, dtype=np.float32),
        'b2': np.zeros(n_words, dtype=np.float32),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / exp.sum(axis=1, keepdims=True)


def forward(contexts: np.ndarray, params: dict[str, np.ndarray]):
    x = params['W1'].T[contexts].mean(axis=1)  # Еквівалент середнього one-hot x @ W1.T
    z1 = x + params['b1']
    hidden = np.maximum(z1, 0)
    logits = hidden @ params['W2'].T + params['b2']
    probabilities = softmax(logits)
    return probabilities, (x, z1, hidden)


def batch_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-12)).mean())


def train_step(params: dict[str, np.ndarray], contexts: np.ndarray,
               targets: np.ndarray, lr: float = .25) -> float:
    """Пряме + ручне зворотне поширення + один мінібатчний крок SGD."""
    probs, (_x, z1, h) = forward(contexts, params)
    loss = batch_loss(probs, targets)
    delta = probs.copy()
    delta[np.arange(len(targets)), targets] -= 1
    delta /= len(targets)
    dW2 = delta.T @ h
    db2 = delta.sum(axis=0)
    dh = delta @ params['W2']
    dz1 = dh * (z1 > 0)
    db1 = dz1.sum(axis=0)
    # Повторені слова накопичують свій градієнт через np.add.at.
    dW1T = np.zeros_like(params['W1'].T)
    np.add.at(dW1T, contexts.ravel(), np.repeat(dz1[:, None, :] / contexts.shape[1],
                                                   contexts.shape[1], axis=1).reshape(-1, dz1.shape[1]))
    params['W2'] -= lr * dW2
    params['b2'] -= lr * db2
    params['W1'] -= lr * dW1T.T
    params['b1'] -= lr * db1
    return loss


def train_cbow(x: np.ndarray, y: np.ndarray, vocab_size: int, *, dim: int = 50,
               steps: int = 500, batch_size: int = 128, lr: float = 0.35,
               seed: int = 42, verbose: bool = True):
    if steps < 1:
        raise ValueError('steps must be >= 1')
    rng = np.random.default_rng(seed)
    p = initialize_model(vocab_size, dim, seed)
    history = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(y), size=min(batch_size, len(y)))
        loss = train_step(p, x[ix], y[ix], lr=lr)
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            history.append({'step': step, 'loss': round(loss, 6)})
            if verbose:
                print(f'{step:4d}/{steps} · Loss: {loss:.4f}')
    return p, history, time.perf_counter() - t0


def get_embeddings(params: dict[str, np.ndarray], how: str = 'average') -> np.ndarray:
    if how == 'average':
        return (params['W1'].T + params['W2']) / 2.0
    if how == 'input':
        return params['W1'].T.copy()
    raise ValueError('how must be input or average')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else 0.0


def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def most_similar(word: str, embeddings: np.ndarray, word2idx: dict[str, int], top_n: int = 5):
    word = word.lower().strip()
    if word not in word2idx:
        return []
    e = normalize_vectors(embeddings)
    scores = e @ e[word2idx[word]]
    scores[word2idx[word]] = -np.inf
    scores[word2idx['<unk>']] = -np.inf
    words = list(word2idx)
    ids = np.argsort(-scores, kind='stable')[:top_n]
    return [(words[i], float(scores[i])) for i in ids if np.isfinite(scores[i])]


def analogy(positive: list[str], negative: list[str], embeddings: np.ndarray,
            word2idx: dict[str, int], top_n: int = 5):
    ws = [w.lower().strip() for w in positive + negative]
    if not positive or any(w not in word2idx for w in ws):
        return []
    v = sum((embeddings[word2idx[w]] for w in positive), start=np.zeros(embeddings.shape[1], dtype=np.float32))
    v -= sum((embeddings[word2idx[w]] for w in negative), start=np.zeros(embeddings.shape[1], dtype=np.float32))
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return []
    similarities = normalize_vectors(embeddings) @ (v / norm)
    for w in set(ws + ['<unk>']):
        similarities[word2idx[w]] = -np.inf
    words = list(word2idx)
    ids = np.argsort(-similarities)[:top_n]
    return [(words[i], float(similarities[i])) for i in ids if np.isfinite(similarities[i])]


def compute_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """PCA через SVD; для діаграм потрібні щонайменше 2 точки."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < n_components or X.shape[1] < n_components:
        raise ValueError('Недостатня кількість точок чи координат для PCA')
    centered = X - X.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:n_components].T


def save_embeddings(path: str | Path, word2idx: dict[str, int], embeddings: np.ndarray, params=None):
    """NPZ без pickle: зберігає слова, матрицю подань та (за наявності) ваги."""
    names = list(word2idx)
    items = {'words': np.array(names), 'embeddings': np.asarray(embeddings, dtype=np.float32)}
    if params:
        items.update({k: v for k, v in params.items()})
    np.savez_compressed(path, **items)


def load_embeddings(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        words = data['words'].tolist()
        emb = data['embeddings'].copy()
    return {w: i for i, w in enumerate(words)}, emb


def create_ui(word2idx: dict[str, int], embeddings: np.ndarray):
    """Запускається в Jupyter / Google Colab; потрібен ipywidgets."""
    import ipywidgets as widgets
    from IPython.display import clear_output, display
    word_field = widgets.Text(value='king' if 'king' in word2idx else '', description='Слово:')
    top_slider = widgets.IntSlider(value=5, min=1, max=20, description='Кількість:')
    similar_btn = widgets.Button(description='Знайти подібні', button_style='info')
    expression = widgets.Text(value='king - man + woman', description='Вираз:', layout=widgets.Layout(width='600px'))
    arith_btn = widgets.Button(description='Обчислити', button_style='info')
    output = widgets.Output()

    def show_similar(_):
        with output:
            clear_output()
            result = most_similar(word_field.value, embeddings, word2idx, top_slider.value)
            print('Подібні слова:' if result else 'Слово відсутнє у словнику.')
            for w, score in result:
                print(f'  {w:18s} cos={score:.4f}')

    def show_analogy(_):
        with output:
            clear_output()
            text = expression.value.lower().replace('-', '+-')
            parts = [p.strip() for p in text.split('+') if p.strip()]
            positive = [s for s in parts if not s.startswith('-')]
            negative = [s[1:].strip() for s in parts if s.startswith('-')]
            missing = [w for w in positive + negative if w not in word2idx]
            if missing:
                print('Невідомі слова:', ', '.join(missing)); return
            result = analogy(positive, negative, embeddings, word2idx, top_slider.value)
            if not result:
                print('Не вдалося визначити результат.'); return
            print('Результат векторної арифметики:')
            for w, score in result:
                print(f'  {w:18s} cos={score:.4f}')

    similar_btn.on_click(show_similar); arith_btn.on_click(show_analogy)
    display(widgets.VBox([
        widgets.HTML('<h3>CBOW — дослідження векторних подань</h3>'),
        widgets.HBox([word_field, top_slider, similar_btn]),
        widgets.HBox([expression, arith_btn]), output]))


def save_artifacts(corpus_path: str | Path, output_dir: str | Path, *, steps: int = 500,
                   dim: int = 50, C: int = 2, min_freq: int = 3,
                   seed: int = 42, lr: float = 0.35, verbose: bool = True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences(corpus_path)
    if len(sentences) < 5:
        raise ValueError('Корпус надто малий — недостатньо речень')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    cut = int(.9 * len(order))
    train_sentences = [sentences[i] for i in order[:cut]]
    test_sentences = [sentences[i] for i in order[cut:]]
    word2idx, counts = make_vocab(train_sentences, min_freq=min_freq)
    train_x, train_y = create_examples(train_sentences, word2idx, C=C)
    test_x, test_y = create_examples(test_sentences, word2idx, C=C)
    print(f'Навчальних рядків: {len(train_sentences)} · Тестових: {len(test_sentences)}')
    print(f'Вхідних слів: {sum(len(s) for s in sentences)} · Словник: {len(word2idx)}')
    print(f'Навчальних прикладів: {len(train_y)} · Тестових: {len(test_y)}')
    p, history, duration = train_cbow(train_x, train_y, len(word2idx), dim=dim, steps=steps,
                                      lr=lr, seed=seed, verbose=verbose)
    embeddings = get_embeddings(p)
    save_embeddings(out / 'word_embeddings.npz', word2idx, embeddings, p)
    w2, e2 = load_embeddings(out / 'word_embeddings.npz')
    assert w2 == word2idx and np.allclose(e2, embeddings)
    # Оцінка чесною відкладеною вибіркою; не використовуємо її для навчання.
    sample_ix = np.random.default_rng(seed + 1).choice(len(test_y), size=min(1000, len(test_y)), replace=False)
    test_probs, _ = forward(test_x[sample_ix], p)
    valid_loss = batch_loss(test_probs, test_y[sample_ix])
    valid_acc = float((test_probs.argmax(axis=1) == test_y[sample_ix]).mean())
    n_total = len(sentences)
    used = [w for w in EXAMPLES if w in word2idx]
    if len(used) < 3:
        used = [w for w in sorted(word2idx, key=lambda w: -counts[w]) if w != '<unk>'][:min(12, len(word2idx) - 1)]
    coords = compute_pca(embeddings[[word2idx[w] for w in used]])
    plt.figure(figsize=(11, 7.2))
    plt.scatter(coords[:, 0], coords[:, 1], s=45, alpha=.8)
    for i, word in enumerate(used):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.xlabel('Головна компонента 1'); plt.ylabel('Головна компонента 2')
    plt.title('CBOW — PCA-візуалізація навчених векторних подань')
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / 'pca_embeddings.png', dpi=165); plt.close()

    plt.figure(figsize=(9.5, 5.5))
    plt.plot([r['step'] for r in history], [r['loss'] for r in history], marker='o', ms=3)
    plt.xlabel('Крок SGD'); plt.ylabel('Cross-entropy loss')
    plt.title('Навчання CBOW — динаміка функції втрат')
    plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out / 'training_loss.png', dpi=165); plt.close()

    similar_data = []
    for word in used[:min(len(used), 15)]:
        for rank, (neighbor, cos) in enumerate(most_similar(word, embeddings, word2idx, 5), 1):
            similar_data.append([word, rank, neighbor, round(cos, 6)])
    with open(out / 'similar_words.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['word', 'rank', 'similar_word', 'cosine_similarity']); writer.writerows(similar_data)
    with open(out / 'pca_coordinates.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['word', 'pc1', 'pc2'])
        for word, xy in zip(used, coords): writer.writerow([word, float(xy[0]), float(xy[1])])
    with open(out / 'training_history.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss']); writer.writeheader(); writer.writerows(history)
    metric = {
        'source_file': Path(corpus_path).name,
        'total_input_sentences': n_total, 'train_sentences': len(train_sentences), 'test_sentences': len(test_sentences),
        'tokens_in_selected_lines': sum(map(len, sentences)), 'vocabulary_size': len(word2idx),
        'train_examples': len(train_y), 'test_examples': len(test_y),
        'embedding_dimension': dim, 'context_each_side': C, 'training_steps': steps,
        'batch_size': 128, 'learning_rate': lr, 'min_frequency': min_freq,
        'seed': seed, 'training_seconds': round(duration, 3),
        'initial_minibatch_loss': history[0]['loss'], 'final_minibatch_loss': history[-1]['loss'],
        'heldout_loss': round(valid_loss, 5), 'heldout_top1_accuracy': round(valid_acc, 5),
        'heldout_examples_evaluated': len(sample_ix),
        'method': 'CBOW, NumPy ручний backprop, ReLU, softmax, крос-ентропія, mini-batch SGD',
        'analysis_note': 'Невеликий корпус та мала кількість кроків обмежують семантичну якість аналогій.'
    }
    with open(out / 'metrics.json', 'w', encoding='utf-8') as f: json.dump(metric, f, ensure_ascii=False, indent=2)
    with open(out / 'REPORT.md', 'w', encoding='utf-8') as f:
        f.write(f'''# Лабораторна робота № 8 — звіт\n\n## Тема та мета\nВекторні подання слів (Word Embeddings). Побудова та дослідження CBOW.\n\n## Дані та методика\nНавчання на локальному корпусі `shakespeare.txt`. Токенізація англомовних слів; для словника використано слова, які зустрічалися щонайменше {min_freq} рази у навчальній частині. Невідомі слова замінено на `<unk>`. Рядки випадково поділено на 90% для навчання і 10% для тесту (seed={seed}). Контекст: {C} слова ліворуч і {C} праворуч.\n\n## Налаштування моделі\n- Архітектура: CBOW зі середнім one-hot контекстом, прихований ReLU, softmax по всьому словнику, крос-ентропія.\n- Реалізація: ручне пряме/зворотне поширення на NumPy, мінібатчний SGD.\n- Розмірність векторів: **{dim}**. Розмір словника: **{len(word2idx)}**.\n- Кроків оптимізації: **{steps}**; швидкість навчання **{lr}**; batch size **128**.\n\n## Результати\n- Речень (рядків) після фільтрації: **{n_total}**; навчальних **{len(train_sentences)}**, тестових **{len(test_sentences)}**.\n- Навчальних вікон: **{len(train_y)}**; тестових: **{len(test_y)}**.\n- Втрата першого мінібатча: **{history[0]['loss']:.4f}**; останнього: **{history[-1]['loss']:.4f}**.\n- Втрата на відкладеній тестовій вибірці ({len(sample_ix)} контекстів): **{valid_loss:.4f}**.\n- Top-1 accuracy передбачення центрального слова на відкладеній вибірці: **{valid_acc:.2%}**.\n- Навчання тривало приблизно **{duration:.2f} с** у поточному середовищі.\n\n## Дослідження векторів\nТаблиця `similar_words.csv` містить найближчі слова за косинусною подібністю; `pca_coordinates.csv` містить двовимірну проєкцію обраних слів. Модель навчається без готових векторів Word2Vec.\n\n![PCA](pca_embeddings.png)\n\n![Динаміка втрат](training_loss.png)\n\n## Збереження\nВектори та навчені ваги збережено у `word_embeddings.npz` (`allow_pickle=False`). У блоці Colab є можливість завантажити їх і виконати перевірку.\n\n## Висновок\nРеалізовано всі компоненти CBOW, отримано й збережено 50-вимірні подання; PCA і косинусна подібність дозволяють дослідити відношення між словами. Відкладена вибірка та динаміка втрат демонструють результати, але якість складних аналогій обмежена обсягом корпусу і числом кроків навчання. Інтерфейс ipywidgets створено для пошуку сусідів і векторної арифметики.\n''')
    make_website(out.parent / 'docs', metric, used, coords, similar_data, word2idx, embeddings)
    return metric


def make_website(docdir: Path, metrics: dict, used: list[str], coords: np.ndarray,
                 similar_data: list, word2idx: dict[str, int], embeddings: np.ndarray):
    """Генерація статичного сайту для GitHub Pages і інтерактивного пошуку слів."""
    import html
    docdir.mkdir(parents=True, exist_ok=True)
    # Зберігаємо всі навчені вектори, щоб арифметика на сайті збігалася з Python.
    data = {'words': list(word2idx), 'vectors': np.round(embeddings, 5).tolist()}
    with open(docdir / 'vectors.json', 'w', encoding='utf-8') as f: json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
    # Картинки site копіюємо до docs (відносні шляхи для GitHub Pages).
    import shutil
    for name in ['pca_embeddings.png', 'training_loss.png']:
        shutil.copy2(docdir.parent / 'results' / name, docdir / name)
    plotdata = [{'word': w, 'x': float(c[0]), 'y': float(c[1])} for w, c in zip(used, coords)]
    meta_json = json.dumps({'metrics': metrics, 'pca': plotdata}, ensure_ascii=False).replace('</', '<\\/')
    page = r'''<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ЛР №8 — CBOW Word Embeddings</title><style>
:root{font:16px/1.6 Inter,system-ui,Arial,sans-serif;color:#16283a;background:#f1f5f9}
*{box-sizing:border-box}main{max-width:1100px;margin:auto;padding:36px 20px 80px}
.hero{background:linear-gradient(110deg,#122b47,#246992);border-radius:24px;padding:38px;color:white;margin-bottom:22px}
.hero h1{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:10px 0}.hero p{max-width:750px;color:#d6e9f4}
.label{letter-spacing:2px;text-transform:uppercase;font-weight:bold;color:#83d9e6;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:20px 0}
.card,.section{background:white;border:1px solid #dce5ee;border-radius:16px;padding:22px}
.metric{font-size:32px;font-weight:800;color:#17618a}.sub{color:#65788c;font-size:13px}
.section{margin:18px 0}.section h2{margin:0 0 8px} .two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
img{width:100%;height:auto;border-radius:9px} input,select,button{font:inherit;padding:11px 12px;border:1px solid #c7d7e3;border-radius:9px}
input{max-width:100%;width:250px}button{background:#17618a;color:white;cursor:pointer;border:0}button:hover{background:#114b6d}
.result{white-space:pre-wrap;background:#f3f8fa;border-radius:12px;padding:15px;min-height:85px;font-family:ui-monospace,monospace;font-size:13px}
footer{color:#66798b;text-align:center;margin-top:24px}@media(max-width:680px){.two{grid-template-columns:1fr}}
</style></head><body><main>
<header class="hero"><div class="label">Лабораторна робота № 8 · NLP</div><h1>Векторні подання слів · CBOW</h1>
<p>Власна реалізація Continuous Bag of Words з NumPy: навчання нейромережі, косинусна подібність, арифметика векторів і візуалізація PCA.</p></header>
<div class="cards" id="metrics"></div>
<section class="section"><h2>Інтерактивна перевірка векторів</h2><p class="sub">Введіть слово з корпусу, щоб знайти найближчі слова. Розрахунок здійснюється у вашому браузері з опублікованими навченими векторами.</p>
<div><input id="query" value="king" aria-label="Слово"><button id="search">Знайти подібні</button></div><pre class="result" id="neighbors">Завантаження векторів…</pre>
<h3>Векторна арифметика</h3><div><input id="expr" value="king - man + woman" style="width:360px" aria-label="Векторний вираз"><button id="calc">Обчислити</button></div><pre class="result" id="arithmetic"></pre></section>
<div class="two"><section class="section"><h2>PCA-візуалізація</h2><img src="pca_embeddings.png" alt="PCA-розташування слів з CBOW"></section>
<section class="section"><h2>Навчання нейромережі</h2><img src="training_loss.png" alt="Графік втрат під час навчання"></section></div>
<section class="section"><h2>Методика і висновки</h2><p>Використано слова з корпусу Shakespeare (англійська мова); CBOW передбачає центральне слово за сусідніми. Вектори мають 50 вимірів. Модель навчена з ReLU, softmax і крос-ентропією; оптимізація — мінібатчний SGD.</p>
<p>Результати на маленькому літературному корпусі не дорівнюють якості великих готових моделей: складні семантичні аналогії можуть бути неточними. Файли з кодом, CSV, вагами та повними результатами доступні в GitHub-репозиторії.</p></section>
<footer>Лабораторна робота №8 · Word Embeddings · Continuous Bag of Words</footer>
</main><script>
const summary=__META__;let model=null, index={};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const m=summary.metrics;
document.querySelector('#metrics').innerHTML=[['Словник',m.vocabulary_size.toLocaleString('uk-UA'),'слів'],['Розмірність',m.embedding_dimension,'компонент'],['SGD',m.training_steps,'кроків'],['Test loss',m.heldout_loss.toFixed(3),'на відкладених даних'],['Top-1', (m.heldout_top1_accuracy*100).toFixed(2)+'%','тестова вибірка']].map(([k,v,foot])=>`<div class="card"><div class="sub">${esc(k)}</div><div class="metric">${esc(v)}</div><div class="sub">${esc(foot)}</div></div>`).join('');
function similarity(query,exclude){let n=Math.hypot(...query);if(!n)return[];let out=[];for(let i=0;i<model.words.length;i++){let w=model.words[i];if(exclude.has(w))continue;let s=0,v=model.vectors[i];for(let j=0;j<v.length;j++)s+=query[j]*v[j];let nv=Math.hypot(...v);out.push([w,nv ? s/(n*nv) : 0])}return out.sort((a,b)=>b[1]-a[1]).slice(0,8)}
function simWord(){if(!model)return;let w=document.getElementById('query').value.trim().toLowerCase(),i=index[w];let r=document.getElementById('neighbors');if(i===undefined){r.textContent='Слова немає у словнику.';return;}r.textContent=similarity(model.vectors[i],new Set([w,'<unk>'])).map(([w,s],i)=>`${i+1}. ${w.padEnd(16,' ')} ${s.toFixed(4)}`).join('\n')}
function calc(){if(!model)return;let expr=document.getElementById('expr').value.trim().toLowerCase(),terms=expr.match(/[+-]?\s*[a-z]+(?:'[a-z]+)?/g)||[],output=document.getElementById('arithmetic');let v=new Array(model.vectors[0].length).fill(0),ex=new Set(['<unk>']);if(!terms.length){output.textContent='Введіть вираз, наприклад king - man + woman';return;}
for(let term of terms){let neg=term.trim().startsWith('-'),w=term.replace(/^[+-]/,'').trim(),ix=index[w];if(ix===undefined){output.textContent=`Невідоме слово: ${w}`;return;}let src=model.vectors[ix];ex.add(w);for(let j=0;j<v.length;j++)v[j]+=(neg?-1:1)*src[j]}output.textContent=similarity(v,ex).map(([w,s],i)=>`${i+1}. ${w.padEnd(16,' ')} ${s.toFixed(4)}`).join('\n')}
document.querySelector('#search').addEventListener('click',simWord);document.querySelector('#calc').addEventListener('click',calc);
fetch('./vectors.json').then(r=>{if(!r.ok)throw new Error('Файл моделі не знайдено');return r.json()}).then(x=>{model=x;x.words.forEach((w,i)=>index[w]=i);simWord();calc()}).catch(e=>{document.getElementById('neighbors').textContent='Помилка завантаження: '+e.message});
</script></body></html>'''
    page = page.replace('__META__', meta_json)
    (docdir / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Лабораторна №8 — CBOW на NumPy')
    parser.add_argument('--corpus', default='shakespeare.txt')
    parser.add_argument('--output', default='results')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--min-freq', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.35)
    args = parser.parse_args()
    metrics = save_artifacts(args.corpus, args.output, steps=args.steps, dim=args.dim, min_freq=args.min_freq, lr=args.lr)
    print('\nОТРИМАНО ФАКТИЧНІ РЕЗУЛЬТАТИ:')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
