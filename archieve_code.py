# Notebook/local-ready version.
# In notebook runtimes, charts can be displayed inline. In local terminal runs,
# charts and selected result tables are written to the results folder.

from pathlib import Path as _Path
import importlib.util as _importlib_util
import json as _json
import os as _os
import sys as _sys
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown
except Exception:
    Markdown = None
    def display(*objs, **kwargs):
        for obj in objs:
            print(obj)

IN_NOTEBOOK = _importlib_util.find_spec("IPython") is not None
try:
    _SCRIPT_DIR = _Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = _Path.cwd()

try:
    import plotly.io as _pio
    _pio.renderers.default = "notebook_connected" if IN_NOTEBOOK else "browser"
except Exception as _exc:
    print(f"Plotly renderer setup skipped: {_exc}")

SHOW_PLOTS = IN_NOTEBOOK or _os.getenv("SHOW_PLOTS", "0").lower() in {"1", "true", "yes"}
SAVE_RESULTS = (not IN_NOTEBOOK) or _os.getenv("SAVE_RESULTS", "0").lower() in {"1", "true", "yes"}
RESULTS_DIR = _SCRIPT_DIR / "results"
_PLOT_COUNTER = 0


def _safe_name(name):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)).strip("_")


def show_plot(fig, name=None):
    global _PLOT_COUNTER
    if SAVE_RESULTS:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _PLOT_COUNTER += 1
        output_name = _safe_name(name or f"plotly_figure_{_PLOT_COUNTER:02d}")
        output_path = RESULTS_DIR / f"{output_name}.html"
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"Saved Plotly chart: {output_path}")
    if SHOW_PLOTS:
        fig.show()
    else:
        print("Plot display skipped. Set SHOW_PLOTS=1 to display charts.")


def show_matplotlib_plot(name=None):
    global _PLOT_COUNTER
    if SAVE_RESULTS:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _PLOT_COUNTER += 1
        output_name = _safe_name(name or f"matplotlib_figure_{_PLOT_COUNTER:02d}")
        output_path = RESULTS_DIR / f"{output_name}.png"
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved Matplotlib chart: {output_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
        print("Matplotlib display skipped. Set SHOW_PLOTS=1 to display charts.")

def _install_result_hooks():
    """Compatibility no-op. Interactive environments display figures via normal show()."""
    return None

def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
    except Exception:
        pass
    return str(obj)

def _save_object(name, obj):
    """Save lightweight result objects when running locally."""
    if not SAVE_RESULTS:
        return None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(name)
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            path = RESULTS_DIR / f"{safe}.csv"
            obj.to_csv(path, index=True)
            print(f"Saved DataFrame: {path}")
            return path
        if isinstance(obj, pd.Series):
            path = RESULTS_DIR / f"{safe}.json"
            path.write_text(_json.dumps(obj.to_dict(), indent=2, default=_json_default), encoding="utf-8")
            print(f"Saved Series: {path}")
            return path
    except Exception:
        pass
    path = RESULTS_DIR / f"{safe}.json"
    path.write_text(_json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved object: {path}")
    return path

def _save_named_results(scope, cell_idx=None):
    """Save selected teaching outputs without dumping large tensors."""
    if not SAVE_RESULTS:
        return None
    names = [
        "current_signal",
        "cnn_signal_table",
        "search_results",
        "ranked_results",
        "best_params",
        "decision_explanations",
        "summary",
        "agentic_context",
        "agentic_llm_report",
    ]
    for name in names:
        if name in scope:
            _save_object(name, scope[name])
    return None


# Feature Engineering
def ohlc_to_culr(ohlc):
    o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    real_body = c - o
    return np.stack([c, upper, lower, real_body], axis=1)


def ts_to_gasf(ts):
    ts = np.asarray(ts, dtype=np.float32)
    lo, hi = np.min(ts), np.max(ts)
    # Match the FinancialVision paper implementation: min-max scale to [0, 1]
    # before applying the Gramian Angular Summation Field transform.
    scaled = np.zeros_like(ts) if np.isclose(lo, hi) else (ts - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def window_to_gaf(window_ohlc, window_volume=None):
    culr = ohlc_to_culr(window_ohlc)
    channels = [ts_to_gasf(culr[:, i]) for i in range(4)]
    
    if window_volume is not None and ADD_VOLUME:
        channels.append(ts_to_gasf(window_volume))
        
    return np.stack(channels, axis=-1)


def _candle_parts(row):
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    body = abs(c - o)
    candle_range = max(h - l, 1e-8)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return o, h, l, c, body, candle_range, upper_shadow, lower_shadow


def label_candlestick_pattern(window_df):
    candles = window_df[["Open", "High", "Low", "Close"]]
    last = candles.iloc[-1]
    o, h, l, c, body, candle_range, upper_shadow, lower_shadow = _candle_parts(last)
    body_ratio = body / candle_range
    recent_close = candles["Close"].tail(min(len(candles), 6)).values.astype(float)
    trend = recent_close[-1] - recent_close[0] if len(recent_close) >= 2 else 0.0

    pattern = "unknown"
    reason = "No 8-pattern rule matched, so the sample is labeled unknown."

    if body_ratio < 0.10:
        pattern = "doji"
        reason = "Open and close are very close relative to the high-low range."
    elif lower_shadow >= 2.0 * max(body, 1e-8) and upper_shadow <= 0.6 * max(body, 1e-8):
        if trend < 0:
            pattern = "hammer"
            reason = "Long lower shadow after a recent decline."
        else:
            pattern = "hanging_man"
            reason = "Long lower shadow after a flat/upward recent move."
    elif upper_shadow >= 2.0 * max(body, 1e-8) and lower_shadow <= 0.6 * max(body, 1e-8):
        pattern = "shooting_star"
        reason = "Long upper shadow with the real body near the low."

    if len(candles) >= 2:
        prev = candles.iloc[-2]
        po, ph, pl, pc, prev_body, prev_range, _, _ = _candle_parts(prev)
        prev_bear = pc < po
        prev_bull = pc > po
        curr_bull = c > o
        curr_bear = c < o
        if prev_bear and curr_bull and o <= pc and c >= po:
            pattern = "bullish_engulfing"
            reason = "Latest bullish body engulfs the previous bearish body."
        elif prev_bull and curr_bear and o >= pc and c <= po:
            pattern = "bearish_engulfing"
            reason = "Latest bearish body engulfs the previous bullish body."

    if len(candles) >= 3:
        c1, c2, c3 = candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]
        o1, h1, l1, cl1, b1, r1, _, _ = _candle_parts(c1)
        o2, h2, l2, cl2, b2, r2, _, _ = _candle_parts(c2)
        o3, h3, l3, cl3, b3, r3, _, _ = _candle_parts(c3)
        midpoint_1 = (o1 + cl1) / 2
        small_middle = b2 / max(r2, 1e-8) < 0.35
        if cl1 < o1 and small_middle and cl3 > o3 and cl3 > midpoint_1:
            pattern = "morning_star"
            reason = "Bearish candle, small indecision candle, then bullish recovery above midpoint."
        elif cl1 > o1 and small_middle and cl3 < o3 and cl3 < midpoint_1:
            pattern = "evening_star"
            reason = "Bullish candle, small indecision candle, then bearish drop below midpoint."

    bias = pattern_to_bias[pattern]
    return {"pattern": pattern, "label_id": pattern_class_names.index(pattern), "bias": bias, "reason": reason}


def detect_candlestick_pattern(window_df):
    return label_candlestick_pattern(window_df)


def pattern_probs_to_signal_probs(pattern_probs):
    pattern_probs = np.asarray(pattern_probs, dtype=np.float32)
    signal_probs = np.zeros((pattern_probs.shape[0], len(signal_names)), dtype=np.float32)
    for pattern_id, signal_id in enumerate(pattern_bias_ids):
        signal_probs[:, signal_id] += pattern_probs[:, pattern_id]
    return signal_probs


def make_supervised_dataset(df, window=WINDOW, horizon=HORIZON):
    
    ohlc = df[["Open", "High", "Low", "Close"]].values.astype(np.float32)
    volume = df["Volume"].values.astype(np.float32) 
    close = df["Close"].values.astype(np.float32)
    
    X, y, times, future_returns, pattern_rows = [], [], [], [], []
    
    for end in range(window - 1, len(df) - horizon):
        start = end - window + 1
        window_df = df.iloc[start:end + 1][["Open", "High", "Low", "Close"]]
        label_info = label_candlestick_pattern(window_df)
        future_ret = (close[end + horizon] - close[end]) / close[end]
        
        window_vol = volume[start:end + 1] if ADD_VOLUME else None
        X.append(window_to_gaf(ohlc[start:end + 1], window_vol))
        y.append(label_info["label_id"])
        
        times.append(df.index[end])
        future_returns.append(future_ret)
        pattern_rows.append({"time": df.index[end], "pattern": label_info["pattern"], "bias": label_info["bias"], "reason": label_info["reason"]})
        
    return np.array(X), np.array(y), np.array(times), np.array(future_returns), pd.DataFrame(pattern_rows)


X, y, times, future_returns, pattern_label_table = make_supervised_dataset(raw)
print("X shape:", X.shape)
print("label distribution:", dict(zip(*np.unique(y, return_counts=True))))
print(pattern_label_table["pattern"].value_counts())

_save_named_results(globals(), cell_idx=9)



# Archieved model function
def build_financialvision_cnn_model(num_classes=len(pattern_class_names)):
    
    n_channels = 5 if ADD_VOLUME else 4
    
    model = keras.Sequential([
        layers.Input(shape=(WINDOW, WINDOW, n_channels)),  # dynamic
        layers.Conv2D(16, (2, 2), padding="same", strides=(1, 1)),
        layers.Activation("sigmoid"),
        layers.Conv2D(16, (2, 2), padding="same", strides=(1, 1)),
        layers.Activation("sigmoid"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
        layers.Activation("softmax"),
    ])
    return model


def load_pretrained_cnn(model_path=PRETRAINED_CNN_MODEL_PATH):
    model_path = ensure_pretrained_cnn_model(model_path)

    try:
        model = keras.models.load_model(str(model_path), compile=False)
        load_mode = "full Keras model"
    except Exception as exc:
        print("Full model load failed; trying FinancialVision architecture + load_weights:", repr(exc))
        model = build_financialvision_cnn_model()
        model.load_weights(str(model_path))
        load_mode = "weights into FinancialVision architecture"

    input_shape = tuple(model.input_shape[1:])
    output_classes = int(model.output_shape[-1])
    expected_input = (WINDOW, WINDOW, 4)
    if input_shape != expected_input:
        raise ValueError(f"Pretrained CNN input shape {input_shape} does not match expected {expected_input}.")
    if output_classes != len(pattern_class_names):
        raise ValueError(
            f"Pretrained CNN output classes {output_classes} does not match pattern_class_names length "
            f"{len(pattern_class_names)}. Check the model/class mapping."
        )

    print(f"Loaded pretrained CNN from {model_path}")
    print(f"Load mode: {load_mode}")
    print("CNN input shape:", model.input_shape, "output shape:", model.output_shape)
    print("Pattern class order used by this demo:", pattern_class_names)
    return model


if USE_PRETRAINED_CNN:
    cnn = load_pretrained_cnn()
else:
    cnn = build_financialvision_cnn_model()
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("CNN input shape:", cnn.input_shape)
    print("CNN output shape:", cnn.output_shape)

cnn = load_pretrained_cnn()

_save_named_results(globals(), cell_idx=12)
