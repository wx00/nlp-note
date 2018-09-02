##序列标记：
> 给定一个 token 序列，推断出一个可能性最大的 label 序列
例如：词性标注、命名实体识别

词序列：${\bf{x}}=x_1\cdots x_T$
标记序列：${\bf{y}}=y_1\cdots y_T$
模型选择一个概率最大的标记序列：$$
{\bf{y}}=\arg \max_y p({\bf{y}}|{\bf{x}}) = \arg \max_y p({\bf{x}},{\bf{y}})$$
##隐马尔可夫模型`(HMM hidden markov model)`：
显式变量 ${\bf{x}}$ 与 隐式变量 ${\bf{y}}$
$$
p({\bf{x}},{\bf{y}})=p({\bf{x}}|{\bf{y}})p({\bf{y}}) \\ 
\approx\prod_{t=1}^{T}p(x_t|y_t)p(y_t|y_{t-1}) \\
p({\bf{y}}) \approx \prod_{t=1}^{T}p(y_t|y_{t-1}) \ {\rm{(Markov\ assumption)}} \\
p({\bf{x}}|{\bf{y}}) \approx \prod_{t=1}^{T}p(x_t|y_t)\ {\rm{(Output\ independence)}} $$
形式化描述：

- 隐藏状态集合 $S = s_1, s_2, \cdots, s_N$ 与初始状态 $s_0$
- 转移概率矩阵 $A = \{a_{ij}=p(s_j|s_i)=\frac{C(s_i \to s_j)}{C(s_i)}\}$
- 可能的输出集合 $O = o_1, o_2, \cdots, o_M$ 
- 输出概率矩阵 $B = \{b_{ik}=p(o_k|s_i)=\frac{C(s_i \to o_k)}{C(s_i)}\}$
综上所述，模型的参数就是两个概率矩阵 $A\in R^{N+1\times N}$ 和 $B\in R^{N\times M}$ 
&emsp;
&emsp;
有监督训练方式：
1 将所有语料拼接称一个长度为 $T$ 的序列，得到 ${\rm{x}}$ 与 ${\rm{y}}$
2 求似然函数：$$
a_{ij}=p(s_j|s_i)=\frac{\sum_{t=1}^{T}[y_{t-1}y_t=s_is_j]}{\sum_{t=1}^{T}[y_t=s_i]} \\ 
b_{ik}=p(o_k|s_i)=\frac{\sum_{t=1}^{T}[x_t=o_k,y_t=s_i]}{\sum_{t=1}^{T}[y_t=s_i]}$$


无监督训练方式（需要有一个已经训练好的模型）：

- E-step：使用模型对数据进行标记$$p(y_{t-1}y_t=s_is_j)$$
- M-step：最大化似然函数 $$
a_{ij}=p(s_j|s_i)=\frac{\sum_{t=1}^{T}p(y_{t-1}y_t=s_is_j)}{\sum_{t=1}^{T}p(y_t=s_i)} \\ 
b_{ik}=p(o_k|s_i)=\frac{\sum_{t=1}^{T}p(x_t=o_k,y_t=s_i)}{\sum_{t=1}^{T}p(y_t=s_i)}$$

##解码问题：
不同的隐藏状态序列 $\bf{y}$ 可能对应相同的输出序列 $\bf{x}$
为了消歧，需要找到 $\bf{x}$ 最有可能对应的 $\bf{y}$，这就是所谓的解码：$$
p({\bf{x}},{\bf{y}})=\prod_{t=1}^{T}p(y_t|y_{t-1})p(x_t|y_t) \\
p(y_t|y_{t-1})\in A\ {\rm{(Transition\ probabilities)}}  \\
p(x_t|y_t)\in B\ {\rm{(Output\ probabilities)}} \\
{\bf{y}}=\arg \max_y p({\bf{y}}|{\bf{x}}) = \arg \max_y p({\bf{x}},{\bf{y}})$$


给定一个输入序列 ${\bf{x}}$ 找到一个 ${\bf{y}}$ 使得 $p({\bf{x}},{\bf{y}})$ 最大化
&emsp;

最简单的方式是使用暴力解法遍历所有可能的 ${\bf{y}}$
算法复杂度大概是 $O(N^T)$，下面使用动态规划来求解这个问题

&emsp;
Viterbi 算法：
子问题：找到长度 $t$ 且状态为 $s$ 的最大概率的子序列 $Q_{t,s}$，其对应的概率为：
$$q_{t,s} = \max_{s'}q_{t-1,s'}\cdot p(s|s')\cdot p(o_t|s)$$
需要的存储空间为 $NT$，每个子问题的时间复杂度为 $O(N^2)$，算法整体复杂度为 $O(TN^2)$

##最大熵马尔可夫模型`(MEMM maximum entropy markov model)`：
显式变量 ${\bf{x}}$ 与 隐式变量 ${\bf{y}}$
$$
p({\bf{y}}|{\bf{x}})=\prod_{t=1}^{T}p(y_t|y_{t-1},x_t) \\
p(y_t|y_{t-1},x_t) = \\ \frac{1}{Z_t(y_{t-1},x_t)}\exp\left(\sum_{k=1}^{K}\theta_kf_k(y_t,y_{t-1},x_t)\right) $$
形式化描述：

- 参数： $\theta_k$
- 特征： $f_k(y_t,y_{t-1},x_t)$
- 正则化项： $\frac{1}{Z_t(y_{t-1},x_t)}$

##条件随机场`(CRF conditional random field)`：
显式变量 ${\bf{x}}$ 与 隐式变量 ${\bf{y}}$
$$
p({\bf{y}}|{\bf{x}})=\frac{1}{Z(x)}\prod_{t=1}^{T} \exp\left(\sum_{k=1}^{K}\theta_kf_k(y_t,y_{t-1},x_t)\right) $$
形式化描述：

- 参数： $\theta$
- 特征： $f(y_t,y_{t-1},x_t)$
- 正则化项： $\frac{1}{Z(x)}$

##特征工程
特征 $f(y_t,y_{t-1},x_t)$ 涉及到：

- 前一标记：$y_{t-1}$
- 当前标记：$y_{t}$
- 当前输出：$x_{t}$

具有这样结构的特征称为`标记-观测特征(Label-observation features)`，特征可以分为两部分：标记部分`(label part)` 与 观测部分`(observation part)`，假设 $y$ 和 $y'$ 分别是当前样本与前一个样本的标签：

- $f(y_t,y_{t-1},x_t)=[y_t=y]g_m(x_t)$
- $f(y_t,y_{t-1},x_t)=[y_t=y][y_{t-1}=y']$
- $f(y_t,y_{t-1},x_t)=[y_t=y][y_{t-1}=y']g_m(x_t)$

其中包含 $y$ 和 $y'$ 的部分就是标记部分，剩下的就是观测部分，也被称为观测函数`(observation function)`，其本质也是一个指示器，输出为0或1，具体实现多种多样，比如：

- 与某个单词相等：$[x_t=v] \qquad \forall v \in V$
- 单词是停用词：$[x_t\in {\rm{stop\ words}}]$
- 单词匹配某种模式：$[{\rm{match}}(x_t, {\rm{pattern}})]$
- 单词具有某个标记：$[s \in tags(x_t)] \qquad \forall s \in S$

值得注意的是，$x_t$ 并不一定对应一个单词，也可以是一个单词序列，比如：$x_t=w_{t-1}w_tw_{t+1}$，这给实现带来了很大的灵活性













