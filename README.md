# SpatialGlobalAveragePooling
安装以及介绍请参考 [www.wangqingbaidu.cn](http://www.wangqingbaidu.cn/article/dltb1478596035.html) and [https://zhuanlan.zhihu.com/p/21550685](https://zhuanlan.zhihu.com/p/21550685) 

``` lua
require ‘nn’
l = nn.SpatialGlobalAveragePooling():cuda()
```

使用中切<font color=red>记</font>使用 `:cuda()`进行GPU运算，要不然跑不了（哒溜君比较懒，没有实现CPU的版本，如果没有会出现`SpatialGlobalAveragePooling in CPU is not implemented!`错误）
