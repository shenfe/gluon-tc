# w2v

在comment数据上使用word2vec训练得到的词向量文件comment.vec.dat，来自`text_label/rtc_new/w2v/comment.vec.dat`。

## 文件修改

### 将分隔符统一成空格

执行`sed -i.bak $'s/\t/ /g' comment.vec.dat`。

### 修复存在问题的词向量

第一次发现第84211行的向量维度不一致。

执行`sed '84211q;d' comment.vec.dat`查看发现该行没有词。

执行`sed -i.bak '84211d' comment.vec.dat`删除该行。

继续发现，第180731行（即原180732行），同样问题。



