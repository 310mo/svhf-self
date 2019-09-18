# svhf-self
[Seeing Voices and Hearing Faces](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/) という論文の一部分を実装したもの。 論文では色々と実装して比較しているけど、2つの顔画像と片方の音声を入力してどっちの顔画像の人かを当てさせるネットワークしか検討してないです。

## 使ったデータ
顔画像はプロジェクトページのリンクのうち Cropped Face Images extracted at 1 fps (7.8 GiB uncompressed) からダウンロードしました。

音声は[VoxCeleb Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) の VoxCeleb1 の Audio files のうち Dev A のみを学習に用いました。また、テストには Test を使用しました。

## データの前処理など
プロジェクトページから顔画像とvoxcelebの対応が書いてあるcsvファイルをダウンロードし、それを元に顔画像と音声のペアを作りました。(mk_fv_pair.py) 論文では、「ペアを作るときは同一動画から切り出したものをペアにしないようにした」とか、「男女やその他諸々が偏っていないか確認した」と書いてあるのですが、そういう配慮はしていないです。(とりあえず Dev A のデータを全部使ってやってみただけ) また、複数ある特定の人物の画像や音声から1つしかデータとして使用していなくて、ここをちゃんとすればもっと大きなデータセットで試せたと思います。

audio_process.pyはスペクトログラムをwavファイルから作成するものです。

mk_pair_data.pyは顔と音声のペアのペアをネットワークの入力用に作成するものです。

## 学習
dataset.pyが上で再作成したデータセットを読み込むもの、net.pyがネットワークを定義したもの、svhf.pyが実際に学習するコードです。Batch Norm層をどれくらい入れるべきかいまいちわからなかったのでConv層と全結合層の前に全部入れてしまっています。

## 結果
論文内では正解率81.0%出ているタスクなんですけど、上の要件で学習したら56.2%しか正解率が出ませんでした。
