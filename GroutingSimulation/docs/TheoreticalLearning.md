# 拉梅常数
+ 第一拉梅常数$\lambda$控制体积变化引起的应力大小（体积弹性响应）。在体积压缩时起主要作用。$$\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}$$
+ 第二拉梅常数$\mu$（剪切模量G）表示材料抵抗剪切变形的能力。越大→越“硬”。$$\mu=\frac{E}{2(1+\nu)}$$
+ $$\sigma = \lambda \cdot tr(\epsilon)\cdot I+2\mu \cdot \epsilon$$
# 张量
+ 张量的核心定义在于坐标变化下的行为
  + 标量：在坐标变换下保持不变的量
  + 向量：在坐标变换下，其分量按照坐标基矢相同的规则进行变换
  + 张量：一个 (m, n) 型的张量，在坐标变换下，其分量由 m 个协变索和 n 个逆变索引组成，并按照特定的、由雅可比矩阵构成的规则进行变换。
  + 从数学/物理本质看，张量是一个不随坐标系改变而改变的几何实体或物理量，其分量在坐标变换下遵循特定的协变规则。
+ 张量的阶数
  + 看索引的数量，有几个索引确定一个元素就是几阶张量。
# 应力张量
+ 在三维状态下，每个方向的应变由以下三部分组成：
  1. 直接应变：由该方向的正应力引起
  2. 泊松应变：由垂直方向的正应力引起
  3. 剪切应变：由剪应力引起
+ 对于正应变分量：
  + $\epsilon_{xx}=\frac{1}{E}[\sigma_{xx}-\nu (\sigma_{yy}+\sigma_{zz})]$
  + $\epsilon_{yy}=\frac{1}{E}[\sigma_{yy}-\nu (\sigma_{xx}+\sigma_{zz})]$
  + $\epsilon_{zz}=\frac{1}{E}[\sigma_{zz}-\nu (\sigma_{yy}+\sigma_{xx})]$
+ 对于剪应变分量：
  + $\epsilon_{xy}=\frac{1}{2G}\sigma_{xy}$
  + $\epsilon_{yz}=\frac{1}{2G}\sigma_{yz}$
  + $\epsilon_{zx}=\frac{1}{2G}\sigma_{zx}$
  + 其中，$G=\frac{E}{2(1+\nu)}$为剪切模量
+ 引入拉梅常数
  + 正应变分量改写：$\sigma_{ii}=\frac{1+\nu}{E}\sigma_{ii}-\frac{\nu}{E}(\sigma_{xx}+\sigma_{yy}+\sigma_{zz})$
  + 表达式为：$\epsilon_{ij}=\frac{1+\nu}{E}\sigma_{ij}-\frac{\nu}{E}\sigma_{kk}\delta_{ij}$
    + $\sigma_{kk}=\sigma_{xx}+\sigma_{yy}+\sigma_{zz}$
    + $\delta_{ij}$为克罗内克符号
  + 定义体积应变：$\epsilon_v=\epsilon_{xx}+\epsilon_{yy}+\epsilon_{zz}$
  + 整理得到：$$\sigma_{ij}=\frac{E}{1+\nu}\epsilon_{ij}+\frac{E\nu}{(1+\nu)(1-2\nu)}\epsilon_v$$
  + 从而得到两个拉梅常数
  + $$\sigma = \lambda \cdot tr(\epsilon)\cdot I+2\mu \cdot \epsilon$$

# 弱形势推导
+ $div(\sigma)+w=0$  引入虚位移$\delta u$并积分
+ 
+ $\int_\Omega[div(\sigma)+w]\cdot\delta u \cdot d\Omega = 0$
+ 
+ $\int_\Omega div(\sigma)\cdot \delta u\cdot d\Omega+\int_\Omega w\cdot \delta u\cdot d\Omega=0$
+ 
+ 根据散度定理有：$\int_S\sigma \cdot \overrightarrow{n} \cdot \delta u \cdot dS - \int_\Omega \sigma:\bigtriangledown(\delta u)\cdot d\Omega+\int_\Omega w\cdot \delta u \cdot d\Omega=0$
+ 
+ 根据边界条件以及$\sigma$的对称性，有$\int_\Omega\sigma:\delta \epsilon \cdot d\Omega=\int_\Omega w\cdot \delta u \cdot d\Omega$
+ 其中，$\sigma=\sigma'+p_wI$
# 切线模量张量 vs 弹性模量张量
  + 切线模量张量：描述材料在当前应力状态下的瞬时应力应变关系，通常是微分形式，依赖于当前应力状态与应力历史
  + 弹性模量张量：描述材料在纯弹性范围内的应力应变关系，只依赖于材料的基本弹性参数。

# 牛顿-拉普森迭代（算法）