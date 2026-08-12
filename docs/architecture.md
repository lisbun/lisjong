# Architecture

## 目的

lisjongは、同じAI PolicyをRiichiEnvでのローカル対局とRiichiLabでの
オンライン対局から利用できるようにする。外部環境のprotocolや型をPolicyから
分離し、観測可能な情報だけを判断へ渡すことを最優先の境界とする。

本書は初期段階の責務と依存方向を固定する。具体的な型やpackage分割は、
RiichiEnvとRiichiLabの実APIを調査した後のIssueで決定する。

## 責務境界

### Policy

Policyは、環境に依存しない観測と合法手の集合を受け取り、選択したactionを返す。

- RiichiEnv、RiichiLab、mjai固有の型や通信処理へ依存しない
- 渡された合法手からだけactionを選択する
- seedを指定した場合に再現可能な判断を行えるようにする
- 非公開情報、完全な山、他家の手牌、内部ゲーム状態を入力として要求しない

### RiichiEnv Adapter

RiichiEnv Adapterは、RiichiEnvとPolicyの間を変換する。

- RiichiEnvの観測と合法手をPolicyの入力へ変換する
- PolicyのactionをRiichiEnvのactionへ変換する
- seatごとの可視性を維持し、変換前後の合法性を検証する
- 対局進行、学習アルゴリズム、Policy固有の判断を所有しない

### RiichiLab Client

RiichiLab Clientは、オンライン接続とsession lifecycleを担当する。

- 認証、接続、受信、送信、再接続、終了処理を担当する
- 受信した公開情報と合法手をPolicy入力へ変換する境界を持つ
- Policyのactionを送信前に再検証する
- tokenをログ、例外、Replay、test fixtureへ含めない
- Policy固有の判断や学習処理を所有しない

## 依存方向

```text
RiichiEnv SDK  →  RiichiEnv Adapter  ┐
                                     ├→  Policy contract  ←  Policy implementation
RiichiLab API  →  RiichiLab Client   ┘
```

矢印は「左側が右側の公開契約を利用する」方向を表す。Policy contractとPolicy
implementationはRiichiEnv SDKおよびRiichiLab APIへ依存しない。外部環境の仕様変更は
AdapterまたはClientで吸収し、Policyへ伝播させない。

## 情報境界

Policyへ渡してよい情報は、そのseatのプレイヤーが判断時点で観測できる情報に限る。

- 自席の手牌と公開済みの牌・宣言・点数
- 判断時点で利用可能な合法手
- 公開ルールと対局進行上必要な公開状態

他家の未公開牌、山の並び、将来のevent、環境内部だけが持つ完全状態は渡さない。
AdapterとClientの変換testでは、値の対応だけでなく禁止情報の欠落も確認する。

## データと秘密情報

model weight、raw牌譜、実験生成物、tokenはsource codeと分離する。外部データや
modelを利用する場合は、提供元、license、version、取得方法、hashを記録する。
秘密情報は環境変数等から実行時に注入し、repositoryへcommitしない。

## 現在の非目標

- RiichiEnv / RiichiLabの具体的な型の確定
- Policy、学習、推論の実装
- Mortalまたはpython-studyとの統合
- Rustによる最適化
- modelや牌譜の取得・配布
