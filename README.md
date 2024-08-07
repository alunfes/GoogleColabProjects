アジェンダ
イントロダクション (10分)

トレーニングの目的と概要
pandasの基本とデータ分析での役割
pandasのインストールとセットアップ (10分)

環境設定
pandasのインストール方法（pip）
基本的なデータ構造 (20分)

SeriesとDataFrameの説明
基本的な操作（作成、表示、インデックス）
データの読み込みと保存 (20分)

CSVファイル、Excelファイル、データベースからのデータ読み込み
データの保存方法（CSV、Excel）
データの操作とクリーニング (30分)

データのフィルタリング、並べ替え、グルーピング
欠損値の処理方法
重複データの削除
データの解析と基本的な統計量 (30分)

データの要約（平均、中央値、標準偏差など）
ピボットテーブルの作成
クロス集計
データの可視化 (20分)

pandasとMatplotlibを使った基本的なグラフ作成
ヒストグラム、棒グラフ、折れ線グラフなど
Q&Aセッション (20分)

受講者からの質問に回答
トラブルシューティング
まとめと次のステップ (10分)

今日の学びの振り返り
次の学習ステップの提案



トレーニングの目的と概要 (5分)
目的:

受講者にpandasの基本を理解させ、データ分析における基礎的な操作を習得させる。
受講者が実際の業務でpandasを使ってデータを処理・分析できるようにする。
概要:

pandasの主要な機能と特長について説明。
データの読み込み、操作、クリーニング、解析、可視化までの一連の流れを紹介。
セッション終了後に得られるスキルや知識を明確にする。
pandasの基本とデータ分析での役割 (5分)
pandasとは:

Pythonで使えるデータ分析用のライブラリであり、データ構造（SeriesとDataFrame）を提供する。
大規模データの操作や解析が容易にできるように設計されている。
データ分析での役割:

データの読み込みと保存:
様々なファイル形式（CSV, Excel, SQLデータベースなど）からデータを読み込み、加工後に保存する機能。
データの操作:
データの選択、フィルタリング、並べ替え、集計などの操作が簡単に行える。
データのクリーニング:
欠損値の処理、重複データの削除、データ型の変換など、データを解析しやすい形に整える機能。
データの解析:
基本的な統計量の計算、ピボットテーブルの作成、クロス集計などの解析機能。
データの可視化:
pandasと他のライブラリ（MatplotlibやSeaborn）を使って、グラフやチャートを作成し、データの視覚的な理解を促進。


Series
概要:

Seriesはpandasの一つのデータ構造で、一次元の配列のようなものです。
ラベル付きデータを格納できるため、インデックスを持つことが特徴です。
特徴:

インデックス:
デフォルトでは0から始まる整数のインデックスを持ちますが、任意のインデックス（ラベル）を指定することも可能です。
データ型:
同じデータ型のデータを格納するため、一貫性があります。



DataFrame
概要:

DataFrameはpandasの最も重要なデータ構造で、二次元の表形式のデータを格納します。
行と列のラベルを持つことができ、異なるデータ型を持つ列を含むことができます。
特徴:

行と列:
インデックス（行ラベル）と列ラベルを持ちます。
データ型:
各列が異なるデータ型を持つことができます。




import pandas as pd

def classify_status(row, df_old):
    if row['Engagement ID'] in df_old['Engagement ID'].values:
        old_phase = df_old.loc[df_old['Engagement ID'] == row['Engagement ID'], 'Phase'].iloc[0]
        if row['Phase'] == old_phase:
            return row['Phase']
        else:
            return f"new {row['Phase'].lower()}"
    else:
        return f"new {row['Phase'].lower()}"

def compare_and_classify(df_old, df_new):
    # New Statusカラムを追加
    df_new['New Status'] = df_new.apply(lambda row: classify_status(row, df_old), axis=1)
    
    # 消失したエンゲージメントを特定
    disappeared = df_old[~df_old['Engagement ID'].isin(df_new['Engagement ID'])]
    disappeared['New Status'] = 'disappeared'
    
    # 結果を結合
    result = pd.concat([df_new, disappeared[['Engagement ID', 'Phase', 'New Status']]])
    
    return result

# 使用例
# df_old = pd.read_csv('old_inventory.csv')
# df_new = pd.read_csv('new_inventory.csv')
# result = compare_and_classify(df_old, df_new)
# result.to_csv('classified_inventory.csv', index=False)
