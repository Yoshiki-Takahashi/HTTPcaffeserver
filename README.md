# HTTPCaffeServer

外部からHTTP通信で送られてくるpng画像に対してCaffeを用いた物体検出を行う．
検出方法は3種類用意した．

- cascadeDetection: OpenCVのカスケード顔検出とCaffeの物体識別を組み合わせる．
- fasterRCNN: fasterRCNNを用いた一般物体検出を行う．
- detectNet: DIGITSのDetectNetを用いて，研究室メンバーの顔写真に対する学習済みネットワークで物体検出を行う．

クライアントのコードでは，webカメラから撮影した動画を毎フレームごとにHTTP通信でサーバに送信し，受け取った検出結果を画像上に四角の枠で表示する．
