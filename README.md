# price-prediction
## by Jimmyさん
>## アイテム最適化。
>
>- まずpreprocess.pyでデータを前処理する。その結果はdata.pklになる
>- data.pklを読込み、predict.pyで推定。
>
>predict.pyの一部のコードとdata.pklをlambdaにアップロードしている。
>
>## ロジック
>
>input（配送アイテムデータとユーザーデータ）に対して、data.pklに含まれる過去事例と類似事例を算出する。
>これを割引率ごとに行っており、もっとも利益額のでる割引率が最適な価格とする方法。

## 価格アップデートの話
### 何のためにやっているのか
AIによって購買確率を計算し、販売利益を最大化するためにやっている

### ファイル構成
- `coupon_discount`: 最新モデルはこのディレクトリに入っている（最初クーポン割引用に作ったので名前にクーポンとついているが気にしないでください）
    - `fetch_data_for_coupon.py`: 必要なデータを取得し、`../data/test_for_coupon.csv`として保存する。
    - `fetch_data_for_verification.py`: クーポン最適価格予測用のデータを取得して、`../data/test_for_coupon_verification.csv`として保存する。
    - `optimize_coupon_discount.py`: クーポン最適価格予測用ソースコード
    - `predict_for_coupon.py`: 購買確率予測用の最新のモデル
    - `prepocess_coupon.py`: 前処理用のソースコード
- `data`: データ保存用のディレクトリ
- `models`: モデル保存用のディレクトリ
- `sql`: 生のSQL保存用のディレクトリ
    - `data_for_coupon.sql`: `fetch_data_for_coupon.py`の元となるSQL
    - `item_price_data.sql`: 旧モデル用のSQL
    
### ライブラリのインストール
```bash
pip install -r requirements.txt
```
    
### ローカルでの実行方法
1. 以下のコマンドでカレントディレクトリを移動する。
```bash
$ cd coupon_discount
```

2. 環境変数`DB_USER`, `DB_HOST`, `DB_PWD`, `SSH_USER`(sshのユーザー名), `SSH_KEY_PATH`(秘密鍵のパス), `SSH_PWD`(秘密鍵のパスフレーズ)を設定する。(セキュリティ保護の観点のためハードコーディングはしていない)

3. 以下を実行し、データを取ってくる（重いので時間かかります）
```bash
$ python fetch_data_for_coupon.py
```

以下コマンドでもOK
```bash
DB_HOST=<db_host> DB_USER=<db_user> DB_PWD=<db_password> SSH_USER=<ssh_user> SSH_KEY_PATH=<秘密鍵のパス> SSH_PWD=<秘密鍵のパスフレーズ> python fetch_data_for_coupon.py
```

4. 以下を実行し、データに前処理をかける
```bash
$ python preprocess_coupon.py
```

5. 以下を実行し、学習させる
```bash
$ python predict_for_coupon.py
``` 
生成されたモデルは`../models/model_coupon.h5`に保存される
    
### 本体のソースのある場所
- `SageMaker`: ノートブックインスタンス`price-prediction2`(regionは`ap-northeast-1`)
    - training dataはS3にアップロード済み。バケットは`sagemaker-price-prediction`, keyは`data/data_for_coupon.csv`。
- `Lambda`: `price-prediction`
- `API Gateway`: `price-prediction`
- `air-closet-node-batch`: [`src/tasks/update_price/price_updater.js`](https://github.com/air-closet/air-closet-node-batch/blob/master/src/tasks/update_price/price_updater.js)
    
### どこで動いているのか
- `SageMaker`: 本ソースコードをSageMaker用に移植。実際にコードを見てもらえるとわかると思うが基本的な処理は同じ。エンドポイントは`sagemaker-tensorflow-scriptmode-2019-09-30-03-13-42-942`
- `Lambda`: ここから上記エンドポイントを叩いている
- `API Gateway`: Lambdaと連携してAPIの口を開けている
- `air-closet-node-batch`: ここから一時間に一回、価格更新のAPIを叩いている

### どう動いているのか
air-closet-node-batch → API Gateway → Lambda → Amazon SageMaker

### node-batchリリース方法
batch-releaseに入って、以下コマンドを叩く。
```bash
$ sudo su - ec2-user
$ cd air-closet-node-batch
$ npm run release-price-update
```

リリースがかかる本番インスタンス名は`ac-price-update-batch-prod`

### NNの詳細について
#### NNとは
TL;DR

```
入力に対して，適切な出力をするように学習するモデルです
```


NNは，人間のニューロン（脳細胞）を模したモデルです．
ニューロンは，一定以上の信号が入力されると，自分も信号を出力します．
で，その出力先はまた違うニューロンで...という形で網目状に作用しています．

また，それらは学習をしていて，

- いくつかの入力に対してどれを重要視するか
- どのように判断して出力を決めるか

といったものが改善されて，人間は頭が良くなっていると捉えられます．
#### Input(特徴量)
- `keizoku`: 継続日数
- `item_kbn`: アイテム区分
- `retail_price`: 小売価格
- `member_special_pric`: 会員特別価格
- `plan_id`: プラン
- `fi_count`: お気に入りアイテム数
- `payment_type`: 支払いタイプ
- `sale_discount`: セールの割引
- `ave_purchase_count`: 購入回数
- `age`: 年齢
- `max_coupon_discount`: 持ってるクーポンの最大値
- `max_coupon_limit`: そのクーポンの期限
- `min_coupon_discount`: 持ってるクーポンの最小値
- `min_coupon_limit`: そのクーポンの期限
- `avg_coupon_discount`: 持ってるクーポンの平均値
- `number_of_coupon`: 持ってるクーポンの枚数
- `days_from_start`: 初めてから何日経っているか
- `season`: 季節

#### Output
- `probability of purchasing`: 購買確率

#### ネットワーク構成（MLP）
```
model = Sequential()
model.add(Dense(32, input_dim=33))
model.add(Activation('relu'))
model.add(Dense(2, input_dim=32))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### ファイル間のフロー図
![pic](docs/flow.png)
