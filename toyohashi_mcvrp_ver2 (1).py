################################################################

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
from amplify import FixstarsClient
from amplify import VariableGenerator
from amplify import one_hot
from amplify import einsum
from amplify import less_equal, ConstraintList
from amplify import Poly
from amplify import Model
from amplify import solve
import copy  # オブジェクトのディープコピー用

##############################
# FixStars 有効なトークンを設定
api_token = "AE/mpODs9XWW40bvSyBs9UZVIEoOKWmtgZo"  

# アニーリング実行回数
num_annering = 1

# アニーリング実行時間(mmSec)
time_annering = 10000

##############################
# 対象とする都道府県、市区名(Open Street Mapのロードデータ使用範囲を指定）
state_name = 'Aichi'
city_name = 'Toyohashi'

##############################
# 対象地域のマップ表示中心座標
mapcenter = [34.7691972, 137.3914667]   #豊橋市役所

##############################
# 一人当たりの必要物資重量(Weight of supplies needed per person)
wgt_per = 4.0   # Kg

#########################################
# Streamlit アプリのページ設定
#########################################
st.set_page_config(
    page_title="豊橋市　救援物資配送_最適ルート",  # ブラウザタブタイトル
    page_icon="🗾",  # タブアイコン
    layout="wide"  # ページレイアウトを横幅いっぱいに設定
)

# ──────────── キャッシュ用関数定義 ────────────
@st.cache_data(ttl=3600)
def load_geojson(path):
    return gpd.read_file(path)

@st.cache_data(ttl=3600)
def load_map_graph(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

#################################
# 重い処理のキャッシュ化
#毎回 OSM からグラフを取り直したり、距離行列を再計算したりすると非常に重くなる為。一度計算した結果をキャッシュ化。

@st.cache_resource(show_spinner=False)
def load_graph(place):
    # OSMnx でのグラフ構築をキャッシュ
    return ox.graph_from_place(place, network_type="drive")

@st.cache_data(show_spinner=False)
def build_distance_matrix(G, nodes):
    # 単一始点ダイクストラで全ペア距離をキャッシュ
    matrix = {}
    for u in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, u, weight="length")
        for v in nodes:
            matrix[(u, v)] = lengths.get(v, float("inf"))
    return matrix

# -----------------------------------------------------------------------------
# Streamlit で使用するセッションステート変数の初期化
# Cloud 版では 1 度目のアクセス時に必ず実行できる位置に置く
# -----------------------------------------------------------------------------
for key in [
    "best_tour",
    "best_cost",
    "points",
    "annering_param",
    "num_of_people",
    "shelter_df",
    # "client",   # ← 削除
    # "map_data",   # ← 削除
    "num_shelter",
    "num_transport",
]:
    st.session_state.setdefault(key, None)


#########################################
# streamlit custom css
#########################################
st.markdown(
"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sawarabi+Gothic&display=swap');
    body{
        font-family: "Sawarabi Gothic", sans-serif;
        font-style: normal;
        font-weight: 400;
    }
    .Qheader{
        background:siliver;
    }
    .Qtitle{
        padding-left:1em;
        padding-right:3em;
        font-size:4em;
        font-weight:600;
        color:darkgray;
    }
    .Qsubheader{
        font-size:2em;
        font-weight:600;
        color:gray;
    }
    .caption{
        font-size:1.5em;
        font-weight:400:
        color:gray;
        align:right;
    }
</style>
""",unsafe_allow_html=True
)

####################################

# 地図経路の色指定リスト（ルート表示時に順番に循環）
_colors = [
    "green",
    "orange",
    "blue",
    "red",
    "cadetblue",
    "darkred",
    "darkblue",
    "purple",
    "pink",
    "lightred",
    "darkgreen",
    "lightgreen",
    "lightblue",
    "darkpurple",
]

####################################
# ファイルパス指定
####################################
root_dir = os.getcwd()  # 作業ディレクトリを基準にファイルを読み込む

node_data = "kyoten_geocode.json"        # 拠点データ(JSON)
num_of_people = "number_of_people.csv"  # 被災者数データ(CSV)

toyohashi_geojson = os.path.join(root_dir, "toyohashi.geojson")   # 豊橋市域だけの GeoJSON

route_file = "path_list_toyohashi.json"  # 経路リストデータ(JSON)
Map_Tile = 'https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png'  # 背景地図タイルURL

#################################
# セッションステートに被災者数データを読み込む（初回のみ）
if st.session_state.get("num_of_people") is None:
    # まずファイルパスを作成
    file_path = os.path.join(root_dir, num_of_people)
    try:
        # ファイルパスを唯一の位置引数に、
        # header/names はキーワード引数で渡す
        np_df = pd.read_csv(
            file_path,
            header=None,
            names=["Node", "num"],
            dtype={"Node": str}         # Node を文字列として読み込む
        )
        # ────────── 追加 ──────────
        # CSV から読み込んだ Node 列は文字列化＆余分な空白を除去
        np_df["Node"] = np_df["Node"].astype(str).str.strip()
        # ──────────────────────
    except FileNotFoundError as e:
        st.error(f"{num_of_people} が見つかりません: {e}")
        st.stop()
    st.session_state["num_of_people"] = np_df

# 避難所データ用の初期化
if 'shelter_df' not in st.session_state:
    st.session_state['shelter_df'] = None

# Folium地図表示サイズとズームレベル設定
GIS_HIGHT = 1000
GIS_WIDE = 750
GIS_ZOOM = 12.0

# ポップアップHTMLフォーマット定義
FORMAT_HTML = '<div>【{type}】<br/><b>{name}</b><br/>住所:{address}<div>'


########################################
# ここからFolium を使う表示系関数
########################################

def disp_baseMap(district,center=mapcenter, zoom_start=GIS_ZOOM):
    m = folium.Map(
        location=center,
        tiles=Map_Tile,
        attr='電子国土基本図',
        zoom_start=zoom_start
    )

    # 市境界をジオJSONで点線描画
    folium.GeoJson(
        district,
        style_function=lambda x: {
            'color': 'gray',
            'weight': 2,
            'dashArray': '5, 5'
        }
    ).add_to(m)
    return m

# 全拠点にマーカーを追加して表示する関数
def plot_marker(m, data):
    for _, row in data.iterrows():
        # Node先頭文字判定による色設定
        if row['Node'][0] == 'S':
            icol = 'blue'
        elif row['Node'][0] == 'D':
            icol = 'pink'
        elif row['Node'][0] == 'W':
            icol = 'red'
        elif row['Node'][0] == 'T':
            icol = 'green'
        else:
            icol = 'yellow'
        # マーカー追加
        folium.Marker(
            location=[row['緯度'], row['経度']],
            popup=f"{row['施設名']} / {row['住所']} ({row['拠点種類']})",
            icon=folium.Icon(color=icol)
        ).add_to(m)

# 選択された避難所・配送拠点をレイヤーに分けてマーカー表示(op_data: {'配送拠点': [...], '避難所': [...]}の辞書)
def plot_select_marker(m, data,op_data):
    actve_layer = folium.FeatureGroup(name="開設")
    actve_layer.add_to(m)
    nonactive_layer = folium.FeatureGroup(name="閉鎖/未開設")
    nonactive_layer.add_to(m)

    for _, row in data.iterrows():
        node = row["Node"]
        # 避難所ノード判定
        if node[0] in ("D", "W", "T", "R"):
            if row['Node'] in (op_data['避難所']):
                icol = 'pink'
                layer=actve_layer
            else:
                icol = 'lightgray'
                layer=nonactive_layer
        
        # 配送拠点ノード判定
        elif row['Node'][0] == 'S':
            if row['Node'] in (op_data['配送拠点']):
                icol = 'blue'
                layer=actve_layer
            else:
                icol = 'gray'
                layer=nonactive_layer
        else:
            continue

        # ポップアップHTML生成
        html =FORMAT_HTML.format(name=row['施設名'],address=row['住所'],type=row['拠点種類'])
        popup = folium.Popup(html, max_width=300)
        
        # マーカーを該当レイヤーに追加
        folium.Marker(
            location = [row['緯度'], row['経度']],
            #popup = f"{row['施設名']} / {row['住所']} ({row['拠点種類']})",
            popup = popup,
            icon = folium.Icon(color=icol)
        ).add_to(layer)

# 太線で最適ルートを描画する関数(best_routes: {車両ID: [ノードインデックス,...], ...})
def draw_route(m, G, best_routes, path_df, node_name_list):
    for k, vehicle_route in best_routes.items():
        layer = folium.FeatureGroup(name=f"ルート {k}")
        layer.add_to(m)

        # 各区間をルートジオメトリとして描画
        for iv in range(len(vehicle_route) - 1):
            start_node = node_name_list[vehicle_route[iv]]
            goal_node = node_name_list[vehicle_route[iv + 1]]
            route = path_df[(path_df['start_node'] == start_node) & (path_df['goal_node'] == goal_node)]['route']
            for route_nodes in route:
              route_gdf = ox.graph_to_gdfs(G.subgraph(route_nodes), nodes=False)
              route_gdf.explore(
                  m = layer,  # folium.FeatureGroupオブジェクトを指定
                  color = _colors[k % len(_colors)],
                  style_kwds = {"weight": 10.0, "opacity": 0.5},
              )
    #folium.LayerControl().add_to(m)
    return m

# 細線で最適ルートを描画する関数(draw_route と同様、線の太さのみ変更)
def draw_route_v2(m, G, best_routes, path_df, node_name_list):
    for k, vehicle_route in best_routes.items():
        layer = folium.FeatureGroup(name=f"ルート {k}")
        layer.add_to(m)
        for iv in range(len(vehicle_route) - 1):
            start_node = node_name_list[vehicle_route[iv]]
            goal_node = node_name_list[vehicle_route[iv + 1]]
            route = path_df[(path_df['start_node'] == start_node) & (path_df['goal_node'] == goal_node)]['route']
            for route_nodes in route:
                route_gdf = ox.graph_to_gdfs(G.subgraph(route_nodes), nodes=False)
                route_gdf.explore(
                    m = layer,  # folium.FeatureGroupオブジェクトを指定
                    color = _colors[k % len(_colors)],
                    style_kwds = {"weight": 3.0, "opacity": 0.5},
                )
    #folium.LayerControl().add_to(m)
    return 

# Node ID から施設名を検索して返す補助関数(data: 拠点データ DataFrame, node: 対象ノードID)
def get_point_name(data,node):
   for i,row in data.iterrows():
      if row['Node']== node:
         return row['施設名']

# 地図表示に必要な各種データを読み込んで dict で返す関数(拠点データ, GeoJSON境界, 経路リスト. OSM道路ネットワーク, ベースマップ)
def set_map_data():

    map_data = {}
    
    # 拠点データ    
    try:
        map_data['node_d'] = pd.read_json(
            os.path.join(root_dir, node_data)
        )
    except FileNotFoundError as e:
        st.error(f"{node_data} が見つかりません: {e}")
        st.stop()
    
    # 行政区域GeoJSON（あらかじめ対象市域のみを出力したファイルを直接読み込む）
    try:
        map_data['gep_map'] = gpd.read_file(toyohashi_geojson)
    except Exception as e:
        st.error(f"豊橋市 GeoJSON 読み込み失敗: {e}")
        st.stop()

    # 経路リスト
    try:
        map_data['path_d'] = pd.read_json(
            os.path.join(root_dir, route_file)
        )
    except FileNotFoundError as e:
        st.error(f"{route_file} が見つかりません: {e}")
        st.stop()

   # OSMnx で道路グラフ取得
    place = {"city": city_name, "state": state_name, "country": "Japan"}
    ox.settings.timeout = 180    # OSMnx のリクエストタイムアウトを 180 秒に

    # pickle キャッシュ読み込み or 無ければ取得して保存
    graph_pickle = os.path.join(root_dir, 'toyohashi_drive_graph.pkl')
    if os.path.exists(graph_pickle):
        # pickle があれば読み込む
        with open(graph_pickle, 'rb') as f:
            G = pickle.load(f)
    else:
        # pickle がなければ OSMnx から取得して pickle に保存
        # G = ox.graph_from_place(place, network_type='drive')
        G = load_graph(place)
        with open(graph_pickle, 'wb') as f:
            pickle.dump(G, f)
    map_data['G'] = G

    # ベース地図作成
    map_data['base_map'] = disp_baseMap(map_data['gep_map'] )

    return(map_data)

# 避難所ごとの被災者数（num）をセッションステートから反映更新する関数
def change_num_of_people():
    np_df = st.session_state['num_of_people']
    shelter_df = st.session_state['shelter_df']
   
    for index, row in shelter_df.iterrows():
         node = row['Node']
         num = row['num']
         #np_df.num[np_df.Node == node] = num
         np_df.loc[np_df.Node == node, 'num'] = num
    st.session_state['num_of_people'] = np_df

########################################
# アニーリング周り(以前の関数群)
########################################

# FixstarsClient を初期化し、認証トークンを設定して返す
def start_amplify():
    client = FixstarsClient()
    client.token = api_token    #上記の有効トークン
    
    return client

# one-hot から得たルートシーケンスを重複除去し、戻り値とする(同一ノード連続出現をまとめて削除)
def process_sequence(sequence: dict[int, list]) -> dict[int, list]:
    new_seq = dict()
    for k, v in sequence.items():
        v = np.append(v, v[0])
        mask = np.concatenate(([True], np.diff(v) != 0))
        new_seq[k] = v[mask]
    return new_seq

# one-hot 配列をルートシーケンス辞書に変換する関数: solution.shape == (steps, nodes, vehicles)
def onehot2sequence(solution: np.ndarray) -> dict[int, list]:
    nvehicle = solution.shape[2]
    sequence = dict()
    for k in range(nvehicle):
        sequence[k] = np.where(solution[:, :, k])[1]
    return sequence

# 単一車両で訪問可能な最多拠点数を計算する関数(demand を昇順で累積し、容量内に収まる数を返す)
def upperbound_of_tour(capacity: int, demand: np.ndarray) -> int:
    max_tourable_bases = 0
    for w in sorted(demand):
        capacity -= w
        if capacity >= 0:
            max_tourable_bases += 1
        else:
            return max_tourable_bases
    return max_tourable_bases

# ---------------------------------------------
# 距離行列作成関数の定義（動的計算 & 無限遠置換）
# ---------------------------------------------
# ノード間距離行列を作成する関数(未登録ルートはNaNを設定し、最後に未登録組み合わせがある場合は例外を投げる)
def set_distance_matrix(path_df, node_list, G):
    n = len(node_list)
    distance_matrix = np.zeros((n, n))
    # 経路リストにない組み合わせは動的に計算
    for i, s in enumerate(node_list):
        for j, g in enumerate(node_list):
            if s == g:
                distance_matrix[i, j] = 0.0
                continue

            # 既存経路を探す
            row = path_df[(path_df['start_node'] == s) & (path_df['goal_node'] == g)]
            if not row.empty:
                distance_matrix[i, j] = row['distance'].iloc[0]
                continue

            # なければ OSMnx/NetworkX で最短経路を計算
            try:
                route = nx.shortest_path(G, s, g, weight='length')
                dist = sum(get_route_edge_attributes(G, route, 'length'))
                distance_matrix[i, j] = dist
                # キャッシュとしてpath_dfにも追加
                path_df.loc[len(path_df)] = {
                    'start_node': s,
                    'goal_node': g,
                    'route': [route],
                    'distance': dist
                }
            except Exception as e:
                # それでも失敗したら一時的に無限大を設定
                distance_matrix[i, j] = np.inf
                st.warning(f"経路計算に失敗しました: {s} → {g} ({e})")
    # 無限大を大きな有限値に置換（無限大が混入するとQUBO正規化でNaN発生）
    finite_max = np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
    # 無限距離は「最大全域距離×10」など、適度に大きな値へ置き換え
    distance_matrix[np.isinf(distance_matrix)] = finite_max * 10
    return distance_matrix

# アニーリング用のパラメータをまとめて計算して返す関数
# (distance_matrix: 距離行列, n_transport_base: 配送拠点数, n_shellter: 避難所数, nbase: 全ノード数, nvehicle: 車両台数, capacity: 車両容量, demand: 各ノードの需要（被災者数）)
def set_parameter(path_df, op_data, np_df, G):
    
    annering_param = {}

    # ノードリスト（配送拠点＋避難所）
    re_node_list = op_data['配送拠点'] + op_data['避難所']

    # 距離行列作成
    distance_matrix = set_distance_matrix(path_df, re_node_list, G)
    
    # 基本パラメータ設定
    n_transport_base = len(op_data['配送拠点'])
    n_shellter = len(op_data['避難所'])
    nbase = distance_matrix.shape[0]
    nvehicle = n_transport_base

    # 車両あたり平均訪問拠点数
    avg_nbase_per_vehicle = (nbase - n_transport_base) // nvehicle

    # 需要配列初期化 
    demand = np.zeros(nbase)
    shel_data = op_data['避難所']
    for i in range(nbase - n_transport_base - 1):
        node = shel_data[i]
        #demand[i + n_transport_base] = np_df.iloc[i,1]
        #demand[i + n_transport_base] = np_df[np_df['Node']==node]['num']
        demand[i + n_transport_base] = np_df.loc[np_df.Node==node, 'num'].iloc[0]

    # 容量計算
    demand_max = np.max(demand)
    demand_mean = np.mean(demand[nvehicle:])

    capacity = int(demand_max) + int(demand_mean) * (avg_nbase_per_vehicle)

    # パラメータ辞書に格納
    annering_param['distance_matrix'] = distance_matrix
    annering_param['n_transport_base'] = n_transport_base
    annering_param['n_shellter'] = n_shellter
    annering_param['nbase'] = nbase
    annering_param['nvehicle'] = nvehicle
    annering_param['capacity'] = capacity
    annering_param['demand'] = demand
    annering_param['npeople'] = np_df

    return annering_param

# Amplify モデルを構築して返す関数(・バイナリ変数 x, 目的関数 objective, 制約条件 constraintsを定義し、Model オブジェクトと変数 x を返す)
def set_annering_model(ap):
    gen = VariableGenerator()
    # 車両ごとの最大訪問拠点数を算出
    max_tourable_bases = upperbound_of_tour(ap['capacity'], ap['demand'][ap['nvehicle']:])
    
    # 変数 x の定義: (ステップ数, ノード数, 車両数)
    x = gen.array("Binary", shape=(max_tourable_bases + 2, ap['nbase'], ap['nvehicle']))
    
    # 出発点・終点および他車両ノード訪問禁止の初期設定
    for k in range(ap['nvehicle']):
        if k > 0:
            x[:, 0:k, k] = 0
        if k < ap['nvehicle'] - 1:
            x[:, k+1:ap['nvehicle'], k] = 0
        x[0, k, k] = 1
        x[-1, k, k] = 1
        # 他車両のノード訪問禁止
        x[0, ap['nvehicle']:, k] = 0
        x[-1, ap['nvehicle']:, k] = 0

    # 1回の配送は1拠点ずつ
    one_trip_constraints = one_hot(x[1:-1, :, :], axis=1)
    # 各避難所は1度だけ訪問
    one_visit_constraints = one_hot(x[1:-1, ap['nvehicle']:, :], axis=(0, 2))

    # 容量制約: 走行中の積載重量合計 <= 容量
    weight_sums = einsum("j,ijk->ik", ap['demand'], x[1:-1, :, :])
    capacity_constraints: ConstraintList = less_equal(
        weight_sums,
        ap['capacity'],
        axis=0,
        penalty_formulation="Relaxation",
    )

    # 目的関数: 距離行列を用いた総移動距離最小化
    objective: Poly = einsum("pq,ipk,iqk->", ap['distance_matrix'], x[:-1], x[1:])

    # 制約の合成とスケーリング
    constraints = one_trip_constraints + one_visit_constraints + capacity_constraints
    constraints *= np.max(ap['distance_matrix'])

    model = Model(objective, constraints)

    return model, x

# Amplify を用いてアニーリング実行し、結果を返す関数(num_cal: 解探索試行回数, timeout: タイムアウト（ms）)
def sovle_annering(model, client, num_cal, timeout):
    client.parameters.timeout = timedelta(milliseconds=timeout)
    result = solve(model, client, num_solves=num_cal)
    if len(result) == 0:
        raise RuntimeError("アニーリングに失敗しました。制約を見直してください。")
    return result


########################################
# ここからStreamlit本体
########################################
# ヘッダー表示
#st.markdown('<div class="Qheader"><span class="Qtitle">Q-LOGIQ</span> <span class="caption">Quantum Logistics Intelligence & Quality Optimization  created by WINKY Force</span></div>', unsafe_allow_html=True)
st.markdown('<div class="Qheader"><span class="Qtitle">えるくお</span> <span class="caption">--Emergency Logistics Quantum Optiviser-- Created by WINKY Force</span></div>', unsafe_allow_html=True)

# カラム分割
gis_st, anr_st = st.columns([2, 1])

# Amplify クライアント初期化
if st.session_state.get("client") is None:
    st.session_state["client"] = start_amplify()
client = st.session_state["client"]

# 地図データ初期化
if st.session_state.get("map_data") is None:
    st.session_state["map_data"] = set_map_data()
map_data = st.session_state["map_data"]

# 地図データ取得失敗時
if map_data is None:                                   
    st.error("地図データの読み込みに失敗しました(右下：Manage appからログが確認できます。")
    st.stop()  # 以降の処理を中断

# データ展開
G = map_data['G']

df = map_data['node_d']
# ────────── 追加 ──────────
# JSON 由来の Node 列も文字列化＆余分な空白を除去
df["Node"] = df["Node"].astype(str).str.strip()
# ──────────────────────

path_df = map_data['path_d']
base_map = map_data['base_map']
base_map_copy = copy.deepcopy(base_map)

# 描画リセットフラグ
st.session_state['redraw'] = False

# セッションから値を取得
best_tour = st.session_state['best_tour']
selected_base = st.session_state['points']
np_df = st.session_state["num_of_people"]

# すべての拠点のリストを取得
all_shelter = df[df['Node'].str.startswith('D')| df['Node'].str.startswith('W')|df['Node'].str.startswith('T')|df['Node'].str.startswith('R')]
all_transport = df[df['Node'].str.startswith('S')]


# 右カラムで拠点選択UIを表示
with anr_st:
  st.markdown('<div class="Qsubheader">拠点リスト</div>',unsafe_allow_html=True)
  spinner_container = st.container()
  st.write("開設されている避難所と配送拠点を選んでください")
  # Pill UI で複数選択
  selected_shelter = anr_st.pills("≪避難所≫",all_shelter['施設名'].tolist(),selection_mode="multi")
  selected_transport = anr_st.pills("≪配送拠点≫",all_transport['施設名'].tolist(),selection_mode="multi")
  st.write("『選択完了後、下のボタンを押してください』")

# 選択されたノードIDリスト
selected_shelter_node   = (
    all_shelter[all_shelter["施設名"].isin(selected_shelter)]
      ["Node"]
      .astype(str)
      .str.strip()
      .tolist()
)
selected_transport_node = (
    all_transport[all_transport["施設名"].isin(selected_transport)]
      ["Node"]
      .astype(str)
      .str.strip()
      .tolist()
)

# 選択数が変化したらツアーリセット
num_shelter = len(selected_shelter_node)
num_transport = len(selected_transport_node)

if num_shelter != st.session_state['num_shelter'] or num_transport != st.session_state['num_transport']:
    st.session_state['num_shelter'] = num_shelter
    st.session_state['num_transport'] = num_transport
    best_tour = None
    st.session_state["best_tour"] = best_tour

# 選択拠点情報をセッションに保存
selected_base = {'配送拠点':selected_transport_node,'避難所':selected_shelter_node}
st.session_state['points'] = selected_base

# ルート探索用ノード順リスト
re_node_list = selected_base['配送拠点'] +selected_base['避難所']

# 地図描画エリア
with gis_st:
  if best_tour !=None:
    # 計算結果表示モード
    st.markdown('<div class="Qsubheader">配送最適化-計算結果</div>',unsafe_allow_html=True)
    selected_base = st.session_state['points']
    plot_select_marker(base_map_copy, df,selected_base)
    #re_node_list = selected_base['配送拠点'] + selected_base['避難所']
    base_map_copy = draw_route(base_map_copy, G, best_tour, path_df, re_node_list)

    # ────────────────────────────────────────────────
    # ここから追記：最適経路探索後でも被災者数テーブルを残す
    # ────────────────────────────────────────────────
        if selected_shelter_node:
            with st.expander("被災者数と必要物資量", expanded=False):
                # 1) 元データ取得＆整形
                np_df = st.session_state["num_of_people"].copy()
                np_df["Node"] = np_df["Node"].astype(str).str.strip()

                # 2) 選択避難所リストから表示用 DataFrame を作成
                tmp = pd.DataFrame({
                    "Node": selected_shelter_node,
                    "Name": [get_point_name(df, n) for n in selected_shelter_node],
                })
                merged = tmp.merge(np_df[["Node", "num"]], on="Node", how="left")
                merged["num"] = merged["num"].fillna(0).astype(int)
                merged["demand"] = merged["num"] * wgt_per / 1000.0

                # 3) DataFrame（読み取り専用）を表示
                st.dataframe(
                    merged.rename(columns={
                        "Name":   "避難所",
                        "num":    "避難者数（人）",
                        "demand": "必要物資量（トン）",
                    }),
                    hide_index=True,
                )
# ────────────────────────────────────────────────

  elif selected_base != None:
    st.markdown('<div class="Qsubheader">避難所・配送拠点の設置</div>',unsafe_allow_html=True)
    plot_select_marker(base_map_copy, df,selected_base)
    # 選択された避難所があれば「被災者数テーブル」を表示
    if selected_shelter_node:
        with st.expander("被災者数と必要物資量"):
            # 1) 元データ取得＆Node列文字列化＋strip
            np_df = st.session_state["num_of_people"].copy()
            np_df["Node"] = np_df["Node"].astype(str).str.strip()

            # 2) 選択リストからテーブルを組み立て
            tmp = pd.DataFrame({
                "Node": selected_shelter_node,
                "Name": [get_point_name(df, n) for n in selected_shelter_node]
            })
            # 3) マージ＆欠損は 0 人で埋め
            merged = tmp.merge(np_df[["Node","num"]], on="Node", how="left")
            merged["num"] = merged["num"].fillna(0).astype(int)
            # 4) 必要物資量(トン) を計算
            merged["demand"] = merged["num"] * wgt_per / 1000.0

            # 5) 列の見映えとキーを指定して DataEditor を表示
            edited = st.data_editor(
                merged,
                column_config={
                    "Node":   {"label":"ノード",               "disabled": True},
                    "Name":   {"label":"避難所",             "disabled": True},
                    "num":    {"label":"避難者数（人）"},
                    "demand": {"label":"必要物資量（トン）",   "disabled": True},
                },
                key="shelter_editor"
            )
            # 6) ユーザー編集後の値をセッションに反映
            if edited is not None:
                # np_df の num を置き換え
                for _, row in edited.iterrows():
                    np_df.loc[np_df["Node"] == row["Node"], "num"] = row["num"]
                st.session_state["num_of_people"] = np_df
                st.session_state["shelter_df"]     = edited
    else:
        # まだ避難所が選択されていない場合のガイダンス
        st.info("右側のペインから選択された避難所の避難者数＆必要物資量が表示されます。")

  else:
    st.markdown('<div class="Qsubheader">避難所・配送拠点の設置</div>',unsafe_allow_html=True)

# レイヤーコントロールと地図表示
  folium.LayerControl().add_to(base_map_copy)
  st_folium(base_map_copy, width=GIS_WIDE, height=GIS_HIGHT)

# ───── 選択数チェック （プロトタイプstreamlitクラウドのスペック都合上） ─────
max_nodes = 50
# ここでは既に定義済みのリスト名を使う
total_selected = len(selected_transport_node) + len(selected_shelter_node)
if total_selected > max_nodes:
    st.warning(f"プロトタイプstreamlitクラウドのスペック都合上、配送拠点と避難所の合計は最大{max_nodes}箇所としています。現在{total_selected}箇所選択されています。")
    st.stop()


# 最適経路探索開始ボタン押下時
if anr_st.button("最適経路探索開始", key="btn_optimize_start"):
    with spinner_container:
        with st.spinner("処理中です。しばらくお待ちください..."):
            try:
                # ── 入力チェック ──
                if not selected_shelter_node or not selected_transport_node:
                    anr_st.warning("避難所・配送拠点をそれぞれ1つ以上選択してください")
                    st.stop()

                # ── パラメータ設定・モデル構築 ──
                annering_param = set_parameter(path_df, selected_base, np_df, G)
                model, x = set_annering_model(annering_param)

                # ── アニーリング実行ループ ──
                loop_max = 20
                best_tour = None
                best_obj = None
                for _ in range(loop_max):
                    result = sovle_annering(model, client, num_annering, time_annering)
                    x_values = result.best.values
                    solution = x.evaluate(x_values)
                    sequence = onehot2sequence(solution)
                    candidate_tour = process_sequence(sequence)
                    cost_val = result.solutions[0].objective

                    # 条件に応じて更新(ここでは最初の解を使う例)
                    best_tour = candidate_tour
                    best_obj = cost_val

                    # ループ終了条件
                    if not any(k in best_tour[k][1:-1] for k in range(annering_param['nvehicle'])):
                        break

                # ── 結果整形 ──
                # メートル→キロメートル変換＋小数第1位
                best_obj = round(best_obj / 1000.0, 1)

                # ── セッションステートに保存 ──
                st.session_state["best_tour"] = best_tour
                st.session_state["best_cost"] = best_obj
                st.session_state["annering_param"] = annering_param
                st.session_state["redraw"] = True

                st.success("処理が完了しました！")

            except Exception as e:
                st.error(f"最適経路探索中にエラーが発生しました：{e}")

# ========== 出力 ==========
if st.session_state['best_tour'] !=None:
  annering_param = st.session_state["annering_param"]
  best_obj = st.session_state['best_cost']
  best_tour = st.session_state['best_tour']
  gis_st.write(f"#### 計算結果")
  distance_matrix = annering_param['distance_matrix']
  demand = annering_param['demand']

  node_no = []
  base_list = []
  weight_list = []
  distance_list = []
  node_list = []
  weight_all = 0
  for item in best_tour.items():
     distance = 0
     weight = 0
     p_node = ""
     for i in range(len(item[1])-1):
        it = item[1][i]
        itn = item[1][i+1]
        distance += distance_matrix[it][itn]
        weight += demand[it]
        p_node += f'{get_point_name(df,re_node_list[it])} ⇒ '
     
     it=item[1][len(item[1])-1]
     p_node += f'{get_point_name(df,re_node_list[it])}'
     #r_str=f"ルート{item[0]} (走行距離:{distance/1000:.2f}km/配送量:{weight/1000*wgt_per:.2f}t)  \n【拠点】{p_node}"
     weight_all += weight
     base_list.append(get_point_name(df,re_node_list[it]))
     w_str=f'{weight/1000*4:.2f}t'
     d_str=f'{distance/1000:.2f}km' 
     node_no.append(item[0])
     weight_list.append(w_str)
     distance_list.append(d_str)
     node_list.append(p_node)
     #gis_st.write(r_str)

  result_df = pd.DataFrame({"ノードNo.":node_no,"配送拠点":base_list,"必要物資量":weight_list,"走行距離":distance_list,"巡回順":node_list})
  columnConfig={
                "ノードNo.": st.column_config.Column(width="small"),
                "配送拠点":  st.column_config.Column(width='medium'),
                "必要物資量": st.column_config.Column(width='small'),
                "走行距離": st.column_config.Column(width='small'),
                "巡回順": st.column_config.Column(width='large') 
  }
  gis_st.dataframe(result_df,
               column_config = columnConfig
    )
  all_str = f'総物資量:{weight_all/1000*wgt_per:.2f}t/総距離: {best_obj} km'
  gis_st.write(all_str)

  #best_tour_markdown = "\n".join([f"{key}: {value}" for key, value in best_tour.items()])
  #gis_st.markdown(best_tour_markdown)

if st.session_state['redraw'] != False:
  st.rerun()
