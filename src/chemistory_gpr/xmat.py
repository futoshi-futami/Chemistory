# -*- coding: utf-8 -*-
# R関数のPython等価実装: build_Xmat, extract_from_xyz9,
#                         compute_inverse_perp_distances, compute_distances_to_CH,
#                         combine_Xmat_files
#
# 依存: pandas, numpy
# pip install pandas numpy

import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
import csv

# ------------------------------------------------------------
# compute_inverse_perp_distances(points_df, C3_vec, H3_vec, C6_vec, H6_vec)
#   与えられた Mg/O 点群について、半直線 C3->H3, C6->H6 への
#   垂線距離の「逆数」を計算（射影係数 t<0 は 0 とする）
#   返り値: points_df に inv_dist_L3, inv_dist_L6 を追加した DataFrame
# ------------------------------------------------------------
def compute_inverse_perp_distances(points_df, C3_vec, H3_vec, C6_vec, H6_vec):
    required_cols = {"xa", "ya", "za"}
    if not required_cols.issubset(points_df.columns):
        raise ValueError('points_df に必要なカラム "xa","ya","za" が存在しません')

    C3_vec = np.asarray(C3_vec, dtype=float)
    H3_vec = np.asarray(H3_vec, dtype=float)
    C6_vec = np.asarray(C6_vec, dtype=float)
    H6_vec = np.asarray(H6_vec, dtype=float)

    if not (len(C3_vec) == len(H3_vec) == len(C6_vec) == len(H6_vec) == 3):
        raise ValueError("C3_vec, H3_vec, C6_vec, H6_vec は全て長さ3のベクトルが必要です")

    d3 = H3_vec - C3_vec
    d6 = H6_vec - C6_vec
    eps = np.finfo(float).eps
    if np.dot(d3, d3) < eps:
        raise ValueError("C3_vec と H3_vec が同一座標です")
    if np.dot(d6, d6) < eps:
        raise ValueError("C6_vec と H6_vec が同一座標です")

    d3_norm2 = np.dot(d3, d3)
    d6_norm2 = np.dot(d6, d6)

    X = points_df[["xa", "ya", "za"]].to_numpy(dtype=float)

    # 射影係数 t = ( (P-C)・d ) / |d|^2
    V3 = X - C3_vec
    t3 = (V3 @ d3) / d3_norm2
    V6 = X - C6_vec
    t6 = (V6 @ d6) / d6_norm2

    # 垂線の足
    foot3 = C3_vec + np.outer(t3, d3)
    foot6 = C6_vec + np.outer(t6, d6)

    # 距離
    dist3 = np.linalg.norm(X - foot3, axis=1)
    dist6 = np.linalg.norm(X - foot6, axis=1)

    # t<0 は「後ろ向き」→ 逆距離は0、距離0も0に
    inv3 = np.where((t3 < 0) | (dist3 < eps), 0.0, 1.0 / dist3)
    inv6 = np.where((t6 < 0) | (dist6 < eps), 0.0, 1.0 / dist6)

    out = points_df.copy()
    out["inv_dist_L3"] = inv3
    out["inv_dist_L6"] = inv6
    return out

# ------------------------------------------------------------
# compute_distances_to_CH(df_sub, C0_vec, H3_vec, H6_vec)
#   各点から C0, H3, H6 への距離を計算（列追加）
#   返り値: dist_to_C0, dist_to_H3, dist_to_H6,
#           xpfrom_C0, ypfrom_C0, zpfrom_C0 を追加した DataFrame
# ------------------------------------------------------------
def compute_distances_to_CH(df_sub, C0_vec, H3_vec, H6_vec):
    for v in (C0_vec, H3_vec, H6_vec):
        if len(v) != 3:
            raise ValueError("C0_vec, H3_vec, H6_vec は全て長さ3のベクトルが必要です")

    X = df_sub[["xa", "ya", "za"]].to_numpy(dtype=float)
    C0 = np.asarray(C0_vec, dtype=float)
    H3 = np.asarray(H3_vec, dtype=float)
    H6 = np.asarray(H6_vec, dtype=float)

    dC0 = np.linalg.norm(X - C0, axis=1)
    dH3 = np.linalg.norm(X - H3, axis=1)
    dH6 = np.linalg.norm(X - H6, axis=1)

    # R版に合わせて厳しめのチェック（ゼロ距離は想定外）
    eps = 1e-10
    if np.any(~np.isfinite(dC0)) or np.any(~np.isfinite(dH3)) or np.any(~np.isfinite(dH6)):
        raise ValueError("距離計算で非数値が発生しました")
    if np.any(dC0 < eps):
        raise ValueError("dist_to_C0 がゼロの点があります")
    if np.any(dH3 < eps):
        raise ValueError("dist_to_H3 がゼロの点があります")
    if np.any(dH6 < eps):
        raise ValueError("dist_to_H6 がゼロの点があります")

    out = df_sub.copy()
    out["dist_to_C0"] = dC0
    out["dist_to_H3"] = dH3
    out["dist_to_H6"] = dH6
    # R版は C0差分をコメントアウトして元座標をそのまま入れている
    out["xpfrom_C0"] = out["xa"].astype(float)
    out["ypfrom_C0"] = out["ya"].astype(float)
    out["zpfrom_C0"] = out["za"].astype(float)
    return out

# ------------------------------------------------------------
# extract_from_xyz9(tag)
#   "xyz_9cell_all{tag}.csv" を読み込み:
#   - Mg/O のみ df_MgO を抽出
#   - C1..C6 の重心 C0、C3/H3/C6/H6 の座標を返す
#   返り値: dict( df_MgO, C0_vec0, C3_vec0, H3_vec0, C6_vec0, H6_vec0 )
# ------------------------------------------------------------
def extract_from_xyz9(tag):
    path = f"xyz_9cell_all{tag}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つからない: {path}")

    df = pd.read_csv(path, dtype=str)
    # 必須列チェック
    need = {"AtomA", "xa", "ya", "za"}
    if not need.issubset(df.columns):
        raise ValueError("列が想定外です。必要: AtomA, xa, ya, za")

    # 数値化
    for c in ["xa", "ya", "za"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 元素名抽出（先頭の英字）
    df["elem"] = df["AtomA"].str.extract(r"^([A-Za-z]+)", expand=False)

    # Mg/O 抽出
    df_MgO = df[df["elem"].isin(["Mg", "O"])][["AtomA", "elem", "xa", "ya", "za"]].copy()

    # C重心
    c_rows = df["AtomA"].isin([f"C{i}" for i in range(1, 7)])
    C_coords = df.loc[c_rows, ["xa", "ya", "za"]].to_numpy(dtype=float)
    if C_coords.shape[0] == 0:
        raise ValueError("C1〜C6 が見つかりません。")
    C0_vec0 = np.nanmean(C_coords, axis=0)

    def _get_vec(name):
        sel = df["AtomA"] == name
        if not np.any(sel):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        row = df.loc[sel, ["xa", "ya", "za"]].iloc[0].to_numpy(dtype=float)
        return row

    C3_vec0 = _get_vec("C3")
    C6_vec0 = _get_vec("C6")
    H3_vec0 = _get_vec("H3")
    H6_vec0 = _get_vec("H6")

    return dict(
        df_MgO=df_MgO.reset_index(drop=True),
        C0_vec0=C0_vec0,
        C3_vec0=C3_vec0,
        H3_vec0=H3_vec0,
        C6_vec0=C6_vec0,
        H6_vec0=H6_vec0,
    )

# ------------------------------------------------------------
# build_Xmat(tag="_a", csvfile=None, xydata=None, xy_shift_file="xy_shift.csv",
#            out=None, outfile=None, verbose=True, filter_adist_first=math.nan)
#   X特徴行列を構築し CSV保存（戻り値: DataFrame）
#   - 入力CSVは既定で xyz_9cell_all{tag}.csv → 無ければ xyz_9cell{tag}.csv
#   - xydata: (x,y[,z]) の DataFrame/ndarray。Noneの時は xy_shift_file 読み
#   - filter_adist_first: 最初ループで adist<閾値 の原子のみ固定選抜（Rと同仕様）
# ------------------------------------------------------------
def build_Xmat(tag="_a",
               csvfile=None,
               xydata=None,
               xy_shift_file="xy_shift.csv",
               out=None,
               outfile=None,
               verbose=True,
               filter_adist_first=math.nan):

    # ---- 入力CSV 決定 ----
    if csvfile is None:
        cand = [f"xyz_9cell_all{tag}.csv", f"xyz_9cell{tag}.csv"]
        csvfile = next((c for c in cand if os.path.exists(c)), None)
        if csvfile is None:
            raise FileNotFoundError("入力CSVが見つかりません: " + " / ".join(cand))
    if verbose:
        print("Using CSV:", csvfile)

    df = pd.read_csv(csvfile, dtype=str)
    need = {"AtomA", "xa", "ya", "za"}
    if not need.issubset(df.columns):
        raise ValueError("列が想定外です。必要: AtomA, xa, ya, za")
    # 数値化
    for c in ["xa", "ya", "za"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Mg/O 抽出
    df["elem"] = df["AtomA"].str.extract(r"^([A-Za-z]+)", expand=False)
    df_MgO = df[df["elem"].isin(["Mg", "O"])][["AtomA", "elem", "xa", "ya", "za"]].copy()
    df_MgO[["xa", "ya", "za"]] = df_MgO[["xa", "ya", "za"]].apply(pd.to_numeric, errors="coerce")

    # C/H から C0, C3, H3, C6, H6
    c_rows = df["AtomA"].isin([f"C{i}" for i in range(1, 7)])
    C_coords = df.loc[c_rows, ["xa", "ya", "za"]].to_numpy(dtype=float)
    if C_coords.shape[0] == 0:
        raise ValueError("C1-C6 が見つかりません")
    C0_vec0 = np.nanmean(C_coords, axis=0)

    def _get_vec(name):
        sel = df["AtomA"] == name
        if not np.any(sel):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return df.loc[sel, ["xa", "ya", "za"]].iloc[0].to_numpy(dtype=float)

    C3_vec0 = _get_vec("C3")
    H3_vec0 = _get_vec("H3")
    C6_vec0 = _get_vec("C6")
    H6_vec0 = _get_vec("H6")

    # ---- shift の取得 ----
    if xydata is not None:
        arr = np.asarray(xydata)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("xydata は 2列(x,y) もしくは 3列(x,y,z) が必要です")
        if arr.shape[1] == 2:
            shift = np.column_stack([arr[:, 0:2], np.zeros(arr.shape[0])])
        else:
            shift = arr[:, 0:3]
    else:
        if not os.path.exists(xy_shift_file):
            raise FileNotFoundError("xy_shift が見つかりません: " + xy_shift_file)
        sh = pd.read_csv(xy_shift_file)
        if sh.shape[1] < 2:
            raise ValueError("xy_shift は 2列(x,y) もしくは 3列(x,y,z) が必要です")
        if sh.shape[1] == 2:
            shift = np.column_stack([sh.iloc[:, 0:2].to_numpy(float),
                                     np.zeros(len(sh))])
        else:
            shift = sh.iloc[:, 0:3].to_numpy(float)

    # ---- 走査して Xmat 構築 ----
    X_rows = []
    selected_index = None

    for i in range(shift.shape[0]):
        add = shift[i, :].astype(float)

        C0_vec = C0_vec0 + add
        C3_vec = C3_vec0 + add
        H3_vec = H3_vec0 + add
        C6_vec = C6_vec0 + add
        H6_vec = H6_vec0 + add

        df_feat = compute_inverse_perp_distances(df_MgO, C3_vec, H3_vec, C6_vec, H6_vec)
        df_dist = compute_distances_to_CH(df_MgO, C0_vec, H3_vec, H6_vec)

        X = pd.DataFrame({
            "atom": df_feat["AtomA"].astype(str).values,
            "invd_LH3": df_feat["inv_dist_L3"].astype(float).values,
            "invd_LH6": df_feat["inv_dist_L6"].astype(float).values,
            "adist": df_dist["dist_to_C0"].astype(float).values
        })

        # 最初のループで adist < 閾値 の原子集合を固定（R仕様）
        if selected_index is None:
            selected_index = np.arange(X.shape[0])
            if np.isfinite(filter_adist_first):
                selected_index = np.where(X["adist"].to_numpy(float) < float(filter_adist_first))[0]
                if selected_index.size == 0:
                    raise ValueError("しきい値に合う原子がありません")
                if verbose:
                    print(f"filtered atoms: {selected_index.size} of {X.shape[0]}")

        Xsub = X.iloc[selected_index].reset_index(drop=True)
        # 列名の構築順は R と同じ
        names_LH3 = (Xsub["atom"] + "_invd_LH3").tolist()
        names_LH6 = (Xsub["atom"] + "_invd_LH6").tolist()
        names_D0  = (Xsub["atom"] + "_dist").tolist()
        feature_names = names_LH3 + names_LH6 + names_D0

        Xnum = pd.concat([Xsub["invd_LH3"], Xsub["invd_LH6"], Xsub["adist"]], axis=0).to_numpy()
        # ↑ concat だと縦結合なのでベクトルの並びに注意 → 下で reshape
        # 正しい並びで行ベクトルを作る
        vec = np.concatenate([
            Xsub["invd_LH3"].to_numpy(float),
            Xsub["invd_LH6"].to_numpy(float),
            Xsub["adist"].to_numpy(float)
        ])
        row = pd.DataFrame([vec], columns=feature_names)
        X_rows.append(row)

    Xmat = pd.concat(X_rows, axis=0, ignore_index=True)

    # ---- 保存 ----
    pick = out if out is not None else outfile
    if pick is None:
        pick = f"Xmat_{tag.lstrip('_')}.csv"
    if not re.search(r"\.csv$", pick, flags=re.IGNORECASE):
        pick = pick + ".csv"
    Xmat.to_csv(pick, index=False)
    if verbose:
        print(f"→ 書き出し: {pick}")
    return Xmat

# ------------------------------------------------------------
# combine_Xmat_files(files, mode=("intersect"|"union_na"),
#                    out=None, add_source=True, source_col="tag")
#   Xmat_*.csv を縦結合
#   - mode="intersect": 共通列のみで結合
#   - mode="union_na" : 列の和集合で結合（不足は NA）
#   out=None なら保存しない、返り値は DataFrame
# ------------------------------------------------------------
def combine_Xmat_files(files,
                       mode="intersect",
                       out=None,
                       add_source=True,
                       source_col="tag"):
    mode = mode.lower()
    if mode not in ("intersect", "union_na"):
        raise ValueError('mode は "intersect" か "union_na" を指定')

    files = [str(f) for f in files if os.path.exists(str(f))]
    if len(files) == 0:
        raise FileNotFoundError("結合対象ファイルがありません。")

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        # 数値は予測前に as.numeric すればOK。ここでは列名保持を優先。
        if add_source:
            # "Xmat_*.csv" からタグを推定
            m = re.match(r"^Xmat_?(.+?)\.[Cc][Ss][Vv]$", os.path.basename(f))
            tg = m.group(1) if m else os.path.basename(f)
            df[source_col] = tg
        dfs.append(df)

    if mode == "intersect":
        common_cols = set(dfs[0].columns)
        for d in dfs[1:]:
            common_cols &= set(d.columns)
        if not common_cols:
            raise ValueError("共通列がありません。")
        # 列順は最初のDFの順序に合わせて固定
        cols = [c for c in dfs[0].columns if c in common_cols]
        res = pd.concat([d[cols] for d in dfs], axis=0, ignore_index=True)
    else:
        # union_na
        all_cols = []
        for d in dfs:
            for c in d.columns:
                if c not in all_cols:
                    all_cols.append(c)
        dfs_pad = []
        for d in dfs:
            miss = [c for c in all_cols if c not in d.columns]
            for m in miss:
                d[m] = np.nan
            dfs_pad.append(d[all_cols])
        res = pd.concat(dfs_pad, axis=0, ignore_index=True)

    if out is not None:
        if not re.search(r"\.csv$", out, flags=re.IGNORECASE):
            out = out + ".csv"
        res.to_csv(out, index=False)
        print(f"→ 書き出し: {out}  ({res.shape[0]} 行 × {res.shape[1]} 列)")

    return res

def make_source_xyz_from_csv(csv_path, angle_deg=10, source_path="source.xyz"):
    """
    元の xyz_9cell_all_XX.csv の末尾12行から
      C6, C1, C2, C3, C4, C5, H1..H6
    の順で source.xyz を作る（Cプログラムが期待する並びにする）
    """
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    atom_rows = rows[-12:]  # ここにC,Hがある前提

    # label -> (elem, x, y, z)
    label2data = {}
    for r in atom_rows:
        if len(r) < 5:
            continue
        label = r[0].strip()
        elem  = r[1].strip()
        x, y, z = r[2].strip(), r[3].strip(), r[4].strip()
        label2data[label] = (elem, x, y, z)

    carbon_order   = ["C6", "C1", "C2", "C3", "C4", "C5"]
    hydrogen_order = ["H1", "H2", "H3", "H4", "H5", "H6"]

    missing = [lab for lab in (carbon_order + hydrogen_order) if lab not in label2data]
    if missing:
        raise KeyError(f"必要なラベルがありません: {missing}")

    with open(source_path, "w", encoding="utf-8") as f:
        f.write("12\n")
        f.write(f"{angle_deg}\n")
        for lab in carbon_order:
            elem, x, y, z = label2data[lab]
            f.write(f"{elem} {x} {y} {z}\n")
        for lab in hydrogen_order:
            elem, x, y, z = label2data[lab]
            f.write(f"{elem} {x} {y} {z}\n")


def write_rotated_csv_copy(csv_path, angle_deg=10, out_csv_path=None):
    """
    rotate_xyz.exe が出した `<angle>-altered.xyz` を読んで，
    元csvの末尾12行だけ回転後座標で置き換えた“別名のCSV”を作る。
    元のCSVは上書きしない。
    """
    csv_path = Path(csv_path)
    if out_csv_path is None:
        # xyz_9cell_all_10.csv → xyz_9cell_all_10_10.csv みたいな名前にする
        out_csv_path = csv_path.with_name(f"{csv_path.stem}_{angle_deg}.csv")

    # 1) 回転後xyzを読む
    altered_xyz = Path(f"{angle_deg}-altered.xyz").read_text(encoding="utf-8").splitlines()
    # 1行目=12, 2行目=角度, 3行目以降= C1..C6,H1..H6
    rotated_lines = altered_xyz[2:]
    rotated = {}
    for ln in rotated_lines:
        parts = ln.split()
        label = parts[0].strip()
        x, y, z = parts[1], parts[2], parts[3]
        rotated[label] = (x, y, z)

    # 2) 元のCSVを読む
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    start = len(rows) - 12
    # 並べ替えたときの“逆対応”で戻す
    # 元の並び: C1,C2,C3,C4,C5,C6,H1..H6
    # sourceに渡した並び: C6,C1,C2,C3,C4,C5,H1..H6
    # なのでこう対応させる
    orig2rotlabel = {
        "C1": "C2",
        "C2": "C3",
        "C3": "C4",
        "C4": "C5",
        "C5": "C6",
        "C6": "C1",
        "H1": "H1",
        "H2": "H2",
        "H3": "H3",
        "H4": "H4",
        "H5": "H5",
        "H6": "H6",
    }

    for i in range(start, len(rows)):
        row = rows[i]
        if len(row) < 5:
            continue
        orig_label = row[0].strip()
        if orig_label not in orig2rotlabel:
            continue
        rot_label = orig2rotlabel[orig_label]
        if rot_label not in rotated:
            continue
        x, y, z = rotated[rot_label]
        rows[i][2] = f"{float(x):.10f}" # rows[i][2] = x
        rows[i][3] = f"{float(y):.10f}" # rows[i][3] = y
        rows[i][4] = f"{float(z):.10f}" # rows[i][4] = z
    # 3) 新しい名前で書き出し
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return str(out_csv_path)
