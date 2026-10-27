# 出典と独立性

参照した考え方:

- TypeSafe公式API: https://docs.typesafe.ai/api/system-one
  型・要求・応答の意味だけを相互運用のために参照。SDKの実装は取り込まない。
- 比較モデルの現行情報: https://docs.typesafe.ai/models
- きしだなおき「GemmaをJevのように使ってみる」:
  https://speakerdeck.com/kishida/use-gemma-like-jev
  生成候補の確率を判断に使う考え方と、26B A4Bの速度面での候補性を参照。
  同氏のllama.cppのJevブランチ、jwenv、djevの実装・重みは取得・流用しない。
- 元モデル: https://huggingface.co/google/gemma-4-26B-A4B-it
  Google公式の元重みと標準Transformersランタイムを使用する。

このディレクトリのプロンプト、型処理、候補射影、HTTPサーバー、
テスト、比較計算、スモーク例は今回の依頼のために新規作成する。
PyTorch / Transformers / FastAPI等の汎用ライブラリは利用する。
BF16構成では標準ローダーのaccelerate 1.15.0を使用する。
2026-10-28のFP8候補では、同じARM64汎用vLLMランタイムのLAST poolingを使う。
活性化・正規化を無効にし、公式チェックポイントの26候補行だけを独自コードで射影する。
vLLMの汎用FP8変換を読み込み時に適用し、第三者の派生重みは取得しない。
読み込み用のrunai-model-streamer 0.17.0と純Python依存humanize 4.13.0をDocker内に追加する。
これは汎用の重み読み込み・推論基盤であり、Jev再現コードは含めない。
既存の安全監視の運用閾値を参照するが、Jevの先行再現実装は依存にしない。

比較データは公開testの正解を使用する。先行実装や先行Jevの回答は利用しない。
取得時はHugging Face公式データAPIのx-revisionを固定revisionと照合し、
ラベルごとの割当数に従って先頭から選ぶ。データと出所manifestのSHA256を保存する。

- SNLI: https://huggingface.co/datasets/stanfordnlp/snli
  https://nlp.stanford.edu/projects/snli/ （CC BY-SA 4.0）
- AG News: https://huggingface.co/datasets/fancyzhx/ag_news
  https://arxiv.org/abs/1509.01626 （データ配布元の条件に従う）
- SST-5: https://huggingface.co/datasets/SetFit/sst5
  https://nlp.stanford.edu/sentiment/ （元データの条件に従う）

この3課題は英語の分類・評点の比較であり、全用途、日本語、
画像、複雑な計画、Jev全体の同等性を証明するものではない。
