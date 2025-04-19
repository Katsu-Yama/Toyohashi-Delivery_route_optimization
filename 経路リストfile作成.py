# 必要なライブラリをインポート
import folium
import networkx as nx
import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np
import os
import json

# データファイルのパス設定
dir_name = "/Q-quest2024/teamC/"
root_dir = "/content/drive/MyDrive/" + dir_name
node_data= "Node/kyoten_geocode_Revised.json"

# ノード情報をJSONファイルから読み込み
df=pd.read_json(root_dir+node_data)

# 対象地域の道路ネットワークグラフを取得（車両用）
place = {'city' : 'Odawara',
         'state' : 'Kanagawa',
         'country' : 'Japan'}
G = ox.graph_from_place(place, network_type='drive')

"""
print(G)
ox.plot_graph(G)
"""

# ノードリスト作成：「M」または「K」を含むノードのみを抽出
node_list=pd.concat([df[df['Node'].str.contains('M') ],df[df['Node'].str.contains('K')]], ignore_index=True)

# 距離行列の初期化。ノード数×ノード数のゼロ行列を作成
distance_matrix=np.zeros((len(node_list),len(node_list)))

# 経路情報を格納するリストを初期化。空のリストを作成
path_list=[]

# 各始点ノードに対して処理を繰り返す
for i in range(len(node_list)):
  # 始点ノードの名称・緯度・経度を取得
  start_node_name=node_list.iloc[i]['Node']
  start_lat = node_list.iloc[i]['緯度']
  start_lon = node_list.iloc[i]['経度']
  #print(start_lat)
  #print(start_lon)
  # 始点ノードのOSM（OpenStreetMap）上のノードIDを取得
  start_id = ox.distance.nearest_nodes(G,start_lon,start_lat)

  # 各終点ノードに対して処理を繰り返す
  for j in range(len(node_list)):
    # 終点ノードの名称・緯度・経度を取得
    goal_node_name=node_list.iloc[j]['Node']
    goal_lat = node_list.iloc[j]['緯度']
    goal_lon = node_list.iloc[j]['経度']
    # 終点ノードのOSMノードIDを取得
    goal_id = ox.distance.nearest_nodes(G,goal_lon,goal_lat)

    # 最短経路（ノード列）を計算
    route = ox.shortest_path(G, start_id, goal_id)
    # 最短経路の距離（メートル単位）を計算
    distance = nx.shortest_path_length(G, start_id, goal_id, weight='length')

    # 距離行列に距離を格納
    distance_matrix[i][j]=distance

    # 経路情報を辞書形式で保存
    path={}
    path['m']=i  # 始点インデックス
    path['n']=j  # 終点インデックス
    path['start_node']=start_node_name  # 始点名
    path['goal_node']=goal_node_name  # 終点名
    path['route']=route  # 経路（OSMノードのリスト）
    path['distance']=distance  # 距離（メートル）
           
    # パスリストにパス情報を追加
    path_list.append(path)

    #print(f'start_no.{i}({start_node_name})-goal_no.{j}({goal_node_name})->{distance}')

# 距離行列をCSVファイルに保存
file_path=root_dir+"distance_matrix.csv"
np.savetxt(file_path,distance_matrix,delimiter=",")

# 経路情報リストをJSON形式で保存
file_path=root_dir+"path_list.json"
with open(file_path, 'w') as f:
    json.dump(path_list, f)


