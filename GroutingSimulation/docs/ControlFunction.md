### MC-DP本构模型
> MC-DP模型属于弹性-理想塑性模型，弹性部分采用广义虎克定律，塑性部分采用非关联的流动法则，屈服条件为莫尔-库仑准则，塑性势函数采用Drucker-Prager公式。
#### 屈服函数
+ 在三维主应力空间中定义一个屈服面，当应力点位于屈服面内部时，材料处于弹性状态；当应力点到达屈服面时，材料开始发生塑性变形。
+ 屈服函数$$f = \sqrt{J_2}(\frac{1}{\sqrt{3}}\sin{\theta}\sin{\varphi}+\cos{\theta})-\frac{I_1}{3}\sin{\varphi}-c\cos{\varphi}$$
  + $I_1$为第一应力不变量，$J_2$为第二偏应力不变量，$\theta$为应力洛德角，即$$\sin{3\theta}=-\frac{3\sqrt{3}}{2}\frac{J_3}{J_2^\frac{3}{2}}=-\frac{27J_3}{2q^3}$$q为广义剪应力，$J_3$为第三偏应力不变量。
  + $I_1=\sigma_1+\sigma_2+\sigma_3$
  + $J_2=\frac{1}{6}[(\sigma_1-\sigma_2)^2+(\sigma_2-\sigma_3)^2+(\sigma_3-\sigma_1)^2]$
  + $J_3=(\sigma_1-p)(\sigma_2-p)(\sigma_3-p)$ 其中 $p=\frac{\sigma_1+\sigma_2+\sigma_3}{3}$
#### 塑性势函数与流动法则
+ 当材料屈服后，我们需要一个法则来判断塑性应变发展的方向。这就是流动法则。
+ 它由一个塑性势函数 G 定义。莫尔-库仑模型通常采用非关联流动法则，即塑性势函数 G 与屈服函数 F 不相同。
    + 塑性势函数$$g(I_1,J_2) = \sqrt{J_2}-\alpha I_1=0$$
    + $\alpha$和$\kappa$为模型参数
    + 三轴压缩条件下:$$\alpha=\frac{2\sin{\psi}}{\sqrt{3}(3-\sin{\psi})}$$
    + 流动法则如下：$$d\epsilon_{ij}^p=d\lambda\frac{\partial g}{\partial \sigma_{ij}}$$
#### 硬化/软化定律（本模拟不考虑）
+ 这个定律定义了屈服面在塑性变形过程中如何变化。它描述了塑性应变 ε_pl 与内变量（如粘聚力 c、内摩擦角 φ）之间的关系。
****
#### 本构模型的应用
+ 一致性条件
  + 材料发生塑性变形后，应力点必须始终停留在屈服面上或从屈服面内返回到屈服面上。这个条件数学上表示为：在塑性加载过程中，屈服函数f的值必须保持为0。即$$df=0$$
  + 又因不考虑硬化/软化，一致性条件为$$df=\frac{\partial f}{\partial \sigma_{ij}}d\sigma_{ij}=0$$
+ 应用一致性条件
  + 总应变增量分为弹性与塑性：$$d\epsilon_{ij}=d\epsilon_{ij}^e+d\epsilon_{ij}^p$$
  + 弹性应力由弹性刚度矩阵给出：$$d\sigma_{ij}=D_{ijkl}^ed\epsilon_{kl}^e$$
  + 代入得：$$d\sigma_{ij}=D_{ijkl}^e(d\epsilon_{kl}-d\epsilon_{kl}^p)$$
  + 塑性应变有流动法则给出：$$d\epsilon_{kl}^p=d\lambda\frac{\partial g}{\partial \sigma_{kl}}$$
  + 将上述式子代入一致性条件中：$$\frac{\partial f}{\partial \sigma_{ij}}[D_{ijkl}^e(d\epsilon_{kl}-d\lambda\frac{\partial g}{\partial \sigma_{kl}})]=0$$
+ 求解塑性乘子$d\lambda$
+ 代入$$d\sigma_{ij}=D_{ijkl}^e(d\epsilon_{kl}-d\epsilon_{kl}^p)$$得到真实的应力增量

****
### 平衡方程
+ 由于土骨架的位移变化极其缓慢，认为其加速度为0，因此得到平衡方程：$$div(\sigma'+p_wI)+w=0$$
+ 其中，$\sigma'$为有效应力张量，$p_w$为孔隙水压力，$w$为单位体积所受外力的合力，本问题中为重力。
****
### 位移场求解流程
> 初始化：a₀, σ'₀, 状态变量₀
↓
对于每个载荷步/时间步：
    ↓
    Newton-Raphson迭代 (k = 0, 1, 2, ...)：
        ↓
        1. 从当前位移 a_k 计算应变：ε_k = B · a_k
        ↓
        2. 调用MC-DP本构模型更新应力：
           σ'_{k+1} = MC_DP_Stress_Update(ε_k, σ'_k, 状态变量_k)
        ↓
        3. 计算残差（不平衡力）：
           R_k = F_ext - [∫B^T·(σ'_{k+1} - p_w I)dΩ]
        ↓
        4. 计算切线刚度矩阵：
           K_T,k = ∫B^T · D_T · B dΩ
           （D_T 从本构模型获得）
        ↓
        5. 求解位移增量：K_T,k · Δa = R_k
        ↓
        6. 更新位移：a_{k+1} = a_k + Δa
        ↓
        检查收敛：||R_k|| < tol 且 ||Δa|| < tol
        ↓
        收敛 → 进入下一步
        不收敛 → k = k+1，继续迭代