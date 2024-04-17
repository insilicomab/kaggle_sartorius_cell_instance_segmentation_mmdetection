# Sartorius Cell Instance Segmentation MMDetection

## 依存環境

- Python 3.11.6
- cuda 12.1.1
- pytorch==2.1.1
- mmdetection==3.3.0

## 環境構築(Docker)

### CLI 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up -d
```

2\. コンテナのシェルを起動する

```
docker compose exec -it mmdetection /bin/bash
```

3\. シェルを使ってコマンドを実行する

例）

```
root@xxxxxxxxxx:/workspace# python mmdetection/tools/train.py <config>
```

4\. シェルから抜ける

```
exit
```

### Dev Containers 上で学習を行う場合

1\. コンテナの作成と実行

```
docker compose up
```

2\. リモートエクスプローラーの「開発コンテナー」を選択し、起動したコンテナにアタッチする

3\. VSCode 上でターミナルを表示し、コマンドを実行する

```
root@xxxxxxxxxx:/workspace# python mmdetection/tools/train.py <config>
```

### コンテナの停止

```
docker compose stop
```

再起動する際は以下のコマンドを実行する。

```
docker compose start
```

### コンテナの削除

```
docker compose down
```

## 学習

[公式 Docs](https://mmdetection.readthedocs.io/en/latest/user_guides/train.html)

1\. データ格納先をバンドマウントするため[compose.yaml](./compose.yaml)の`volumes`を設定する

2\. 仮想環境を起動

```
docker compose up -d
```

3\. 学習を実行する

```
python mmdetection/tools/train.py <config>
```

- 引数

  - `config`: トレーニング構成ファイルのパス
  - `--work-dir`: ログとモデルを保存するディレクトリ
  - `--amp`: 自動混合精度トレーニングを有効にする
  - `--auto-scale-lr`: LR の自動スケーリングを有効にする
  - `--resume`: チェックポイントパスを指定すると、それから再開する。指定しない場合は、ワークディレクトリ内の最新のチェックポイントから自動的に再開を試みる。
  - `--cfg-options`: 使用される構成の一部の設定を上書きする。xxx=yyy 形式のキーと値のペアが構成ファイルにマージされる。上書きされる値がリストの場合、`key="[a,b]"`または`key=a,b`のように指定する必要がある。ネストされたリスト/タプルの値も許可されており、例えば`key="[(a,b),(c,d)]"`。引用符が必要であり、空白は許可されていない。
  - `--launcher`: ジョブの起動プログラム。`choices=['none', 'pytorch', 'slurm', 'mpi']`で、デフォルトは`'none'`
  - `--local_rank`: ローカルランク。PyTorch バージョンが 2.0.0 以上の場合、「torch.distributed.launch」は`tools/train.py`に`--local-rank`パラメータを渡すが、それ以外の場合は`--local_rank`パラメータを使用する。デフォルトは 0

- ランダムシード

  - ランダムシードを設定する場合は`--cfg-options randomness.seed=<数字>`を指定する。

例）

```
python mmdetection/tools/train.py config/mask-rcnn/mask-rcnn_r50_fpn_1x_coco_cell.py --cfg-options randomness.seed=0
```

## 評価

[公式 Docs](https://mmdetection.readthedocs.io/en/latest/user_guides/test.html)

1\. データ格納先をバンドマウントするため[compose.yaml](./compose.yaml)の`volumes`を設定する

2\. 仮想環境を起動

```
docker compose up -d
```

3\. 評価を実行する

```
python mmdetection/tools/test.py <config> <checkpoint>
```

- 引数
  - `config`: テスト構成ファイルへのパス
  - `checkpoint`: チェックポイントファイルへのパス
  - `--work-dir`: 評価メトリクスが保存されるディレクトリのパス。存在しないディレクトリの場合は新たに作成される
  - `--out`: オフライン評価用に予測結果を pickle ファイルにダンプするためのパス
  - `--show`: 予測結果を表示する
  - `--show-dir`: 描かれた画像が保存されるディレクトリ。指定された場合、`work_dir/timestamp/show_dir`に自動的に保存される
  - `--wait-time`: 表示のインターバル（秒）。デフォルトは 2 秒
  - `--cfg-options`: 使用される設定ファイルの一部の設定を上書きする。xxx=yyy の形式のキーと値のペアは設定ファイルにマージされる。上書きする値がリストの場合、`key="[a,b]"`または`key=a,b`のように指定する。ネストされたリスト/タプルの値も許容される。例: `key="[(a,b),(c,d)]"`。引用符が必要であり、空白は許可されていない
  - `--launcher`: ジョブの起動方法。`choices=['none', 'pytorch', 'slurm', 'mpi']`で、デフォルトは`'none'`
  - `--tta`: テスト時にテストタイムオーグメンテーション（TTA）を使用する
  - `--local_rank`: ローカルランク。デフォルトは 0

## 推論

[公式 Docs](https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html#image-demo)

1\. データ格納先をバンドマウントするため[compose.yaml](./compose.yaml)の`volumes`を設定する

2\. 仮想環境を起動

```
docker compose up -d
```

3\. 推論を実行する

```
python mmdetection/demo/image_demo.py <inputs> <model>
```

- 引数
  - `inputs`: 入力画像ファイルまたはフォルダのパス
  - `model`: Config ファイルまたはチェックポイントファイル(`.pth`)またはメタファイルで定義されたモデル名とエイリアス。モデル構成ファイルは、パラメータが`.pth`重みファイルの場合、`.pth`から読み込もうとする
  - `--weights`: チェックポイントファイルのパス
  - `--out-dir`: 画像または予測結果の出力ディレクトリ
  - `--device`: 推論に使用するデバイス。デフォルトは`'cuda:0'`
  - `--pred-score-thr`: スコアの閾値。デフォルトは 0.3
  - `--batch-size`: 推論バッチサイズ。デフォルトは 1
  - `--show`: 画像をポップアップウィンドウで表示する
  - `--no-save-vis`: 検出の可視化結果を保存しない
  - `--no-save-pred`: 検出の json 結果を保存しない
  - `--print-result`: 結果を表示するかどうか
  - `--palette`: 可視化に使用されるカラーパレット。`choices=['coco', 'voc', 'citys', 'random', 'none']`で、デフォルトは`'none'`
