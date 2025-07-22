# VideoChat Expert

## TODO 


Codebase实现方面
(基础 detection 版本，测试可行性，验证codebase, 数据现成)
- [x] 在chenting codebase上，添加bbox处理
- 基础mlp实现，embeding回归到bbox，添加额外loss(本周实现并训一版)


（进阶版本，扩增能力）
-  更换decoder，验证decorder的影响 
-  增加 grounding 能力，利用xiangyu数据转换
-  增加 tracking 能力，在之前精细微调数据集的基础上，转换数据
-  增加 Temporal-Spatio Localization能力
-  增加 trajectory prediction 能力

数据方面
目前数据：
- [x] detection (NExT Chat, LLava-G, CogVLM)
-  grounding (xiangyu data转换)
-  tracking (之前收集的detail 数据转换, merlin数据)
-  trajectory prediction
-  Temporal-Spatio Localization

实验验证方面
-  detection 验证代码
-  grounding 验证代码
-  tracking 验证代码
-  trajectory prediction 验证代码
-  Temporal-Spatio Localization 验证代码