# Streamlit Community Cloud 起動停止への対応

## 変更するファイル

1. `toyohashi_mcvrp_streamlitcloud_fixed.py` をリポジトリへ追加する。
2. Streamlit Community Cloud の Main file path を上記ファイルへ変更する。
   - または、既存の `toyohashi_mcvrp_ver2.py` をこの内容で置き換える。
3. `requirements_streamlitcloud_fixed.txt` の内容で、リポジトリ直下の `requirements.txt` を置き換える。

## Amplifyトークン

既存トークンを失効し、新しいトークンを発行する。
Streamlit Community Cloud の App settings > Secrets に次を登録する。

```toml
AMPLIFY_TOKEN = "新しく発行したトークン"
```

トークンをPythonファイルやGitHubへ書かない。

## 修正版の主な変更

- 起動時の `matplotlib`、`geopandas`、`osmnx`、`amplify` の読み込みを廃止。
- `path_list_toyohashi.json` は最適化時または結果描画時だけ読み込む。
- `toyohashi_drive_graph.pkl` は結果描画時だけ読み込む。
- 大きな道路グラフは `st.cache_resource` で全セッション共有する。
- `GeoDataFrame.explore()` を使わず、Foliumの `PolyLine` で経路を描画する。
- 最後の避難所の需要が0になるループ範囲の誤りを修正。
- Cloud向け選択上限を合計20拠点へ変更。

## 再デプロイ

1. GitHubへ変更をpushする。
2. Streamlit Community Cloudで App settings > Secrets を更新する。
3. Reboot app を実行する。
4. 古い依存関係が残る場合はアプリをDeleteして再デプロイする。

## まだ `Killed` が出る場合

初期画面ではなく、最適化完了後のルート描画時に落ちる場合は、`toyohashi_drive_graph.pkl` の展開メモリがCloud上限を超えている可能性がある。その場合は、ローカルで各経路の緯度経度配列を事前生成し、道路グラフpickleを本番アプリから完全に外す。
