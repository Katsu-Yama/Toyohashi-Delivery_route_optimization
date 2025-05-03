
import folium  # 地図描画用ライブラリ
import pandas as pd  # データフレーム操作用ライブラリ
import numpy as np  # 数値計算用ライブラリ
import matplotlib.pyplot as plt  # プロット用ライブラリ（未使用部分あり）
import geopandas as gpd  # 地理データ操作用ライブラリ
import os  # OS関連操作用ライブラリ
import pickle  # グラフ保存・読み込み用ライブラリ
import json  # JSON操作用ライブラリ
import osmnx as ox  # OpenStreetMapデータ取得・操作ライブラリ
import networkx as nx  # グラフ計算用ライブラリ
from geopy.distance import geodesic  # 距離計算用ライブラリ
from datetime import timedelta  # 時間差操作用ライブラリ
from osmnx.utils_graph import get_route_edge_attributes

import streamlit as st  # Streamlitアプリ用ライブラリ
from streamlit_folium import st_folium  # Streamlit上でFolium地図を表示するための関数

# Fixstars Amplify 関係のインポート（量子アニーリング用）
import amplify
from amplify.client import FixstarsClient
from amplify import VariableGenerator
from amplify import one_hot
from amplify import einsum
from amplify import less_equal, ConstraintList
from amplify import Poly
from amplify import Model

# ---------------------------------------------
# グラフデータの読み込み（pickleキャッシュ対応）
# ---------------------------------------------
def load_graph(place, graph_pickle):
    if os.path.exists(graph_pickle):
        with open(graph_pickle, 'rb') as f:
            G = pickle.load(f)
    else:
        G = ox.graph_from_place(place, network_type='drive')
        with open(graph_pickle, 'wb') as f:
            pickle.dump(G, f)
    return G

# ---------------------------------------------
# 距離行列作成関数の定義（動的計算 & 無限遠置換）
# ---------------------------------------------
def set_distance_matrix(path_df, node_list, G):
    n = len(node_list)
    distance_matrix = np.zeros((n, n))
    for i, s in enumerate(node_list):
        for j, g in enumerate(node_list):
            if s == g:
                distance_matrix[i, j] = 0.0
                continue

            row = path_df[(path_df['start_node'] == s) & (path_df['goal_node'] == g)]
            if not row.empty:
                distance_matrix[i, j] = row['distance'].iloc[0]
                continue

            try:
                route = nx.shortest_path(G, s, g, weight='length')
                dist = sum(get_route_edge_attributes(G, route, 'length'))
                distance_matrix[i, j] = dist
                path_df.loc[len(path_df)] = {
                    'start_node': s,
                    'goal_node': g,
                    'route': [route],
                    'distance': dist
                }
            except Exception as e:
                distance_matrix[i, j] = np.inf
                st.warning(f"経路計算に失敗しました: {s} → {g} ({e})")
    finite_max = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
    distance_matrix[np.isinf(distance_matrix)] = finite_max * 10
    return distance_matrix
